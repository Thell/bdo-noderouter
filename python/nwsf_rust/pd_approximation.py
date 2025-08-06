# pd_approximation_best.py
"""
Primal-dual Node-Weighted Steiner Forest approximation solver with bridge heuristics.
"""

# NOTE: It might be interesting to utilized a single sweep over the bridge generated
# cycles to determine which nodes are removable while maintaining connectivity instead
# of solely using the removal set nodes from _removal_candidates which are purely
# based on degree and then passed to the generator for yielding weight based combinations.
#
# A single pass over the cycle basis would only yield individual candidates, but then
# those could be fed to a stable_subgraph to use all_pairs_shortest_paths to get any
# contiguous runs which could then be the candidates for the removal set combinations
# instead of only singletons which can cause massive combinatorial explosion for large
# generated cycles in the mainline of a large tree where the bridge connects several
# long branches.
#
# One of the reasons the removal_set concept was adopted was because you can easily
# remove a node on one side of the main line branch and be ok while you may isntead
# be able to remove several from the other side of the articulation point.

import time
from collections.abc import Generator

import rustworkx as rx
from bidict import bidict
from loguru import logger

import data_store as ds
from api_common import set_logger, ResultDict, SUPER_ROOT
from api_exploration_graph import get_exploration_graph
from api_rx_pydigraph import set_graph_terminal_sets_attribute
from nwsf_rust import IDTree, NodeRouter


class PrimalDualNWSF:
    """Solves Node-Weighted Steiner Forest using primal-dual and bridge heuristics."""

    def __init__(self, exploration_graph: rx.PyDiGraph, terminals: dict[int, int], config: dict) -> None:
        self._logger_level = logger._core.min_level  # type: ignore
        self._do_trace = self._logger_level <= 5
        self._do_debug = self._logger_level <= 10
        self._do_info = self._logger_level <= 20
        self._debug_iter = 0
        if self._do_debug:
            logger.debug("Initializing solver...")

        if self._do_debug:
            self._init_input_validation(exploration_graph, terminals)

        # Testing
        self.testing_orig_terminals = terminals.copy()

        self.config: dict = config
        self.ref_graph: rx.PyGraph = exploration_graph.to_undirected(multigraph=False)
        self.dtree: IDTree = IDTree({v: [] for v in self.ref_graph.node_indices()})
        self.dtree_active_indices: set[int] = set()
        self.bridge_start_time = 0

        ## Mappings
        self.index_to_neighbors: dict[int, set[int]] = {
            v: set(self.ref_graph.neighbors(v)) for v in self.ref_graph.node_indices()
        }
        self.index_to_waypoint: dict[int, int] = dict({
            i: exploration_graph[i]["waypoint_key"] for i in exploration_graph.node_indices()
        })
        self.index_to_weight: dict[int, int] = {
            i: self.ref_graph[i]["need_exploration_point"] for i in self.ref_graph.node_indices()
        }

        # SUPER_ROOT roots denote super terminals with connectivity to any connected base town.
        tmp: bidict[int, int] = exploration_graph.attrs["node_key_by_index"]
        self.terminal_to_root: dict[int, int] = {
            tmp.inv[t]: tmp.inv.get(r, SUPER_ROOT) for t, r in terminals.items()
        }

        self.base_towns: set[int] = set({
            i for i in self.ref_graph.node_indices() if self.ref_graph[i]["is_base_town"]
        })

        # Used in the _combinations_with_weight_range_generator to limit combo generation.
        self.max_node_weight = max(self.index_to_weight.values())

        # Untouchables consists of all terminals, fixed roots, leaf terminal parents
        self.untouchables = self._generate_untouchables()

        # Used in approximation to reduce violated set connectivity checks.
        self.connected_pairs: set[tuple[int, int]] = set()

        # Bridge heuristics

        # 10_000 is the single reverse pass threshold used for benching since early development
        # min 350 => 1.5x the max iter of test cases for 2-direction pass
        # min 125 => 1.5x for reverse sort
        # self.max_removal_attempts = 350
        self.max_removal_attempts = 350

        # Used for controlling the sort reverse flag in the combo generator.
        self._combo_gen_direction = True  # default value reverse=

        # Used to limit the expansion of bridging spanners into the wild frontier
        self.max_bridging_depth = 10

        # Used in reverse deletion to filter deletion and connection checks.
        self._bridge_affected_base_towns: set[int] = set()
        self._bridge_affected_indices: set[int] = set()
        self._bridge_affected_terminals: set[tuple[int, int]] = set()  # (terminal, root) pairs

        if self._do_info:
            logger.info(
                f"Graph nodes: {self.ref_graph.num_nodes()}, Edges: {self.ref_graph.num_edges()}, "
                f"Terminals: {len(self.terminal_to_root)}, "
                f"Super terminals: {len([t for t in self.terminal_to_root.values() if t == SUPER_ROOT])}"
            )

    def _init_input_validation(self, exploration_graph: rx.PyDiGraph, terminals: dict):
        tmp: bidict[int, int] = exploration_graph.attrs["node_key_by_index"]

        assert SUPER_ROOT not in tmp and SUPER_ROOT not in tmp.inv and SUPER_ROOT not in terminals, (
            f"SUPER_ROOT {SUPER_ROOT} is allowed in terminals but not in the graph!"
        )

        # All terminals must be present.
        assert all(t in tmp.inv for t in set(terminals.keys()) | set(terminals.values()) - {SUPER_ROOT}), (
            "All terminals must be present in the graph!"
        )

        # The input exploration graph is a PyDiGraph because the exploration
        # data enforces that structure by having bi-directional links.
        # We use the PyGraph structure here for simplicity and performance of
        # connected component handling which is a core concept of this solver.
        assert isinstance(exploration_graph, rx.PyDiGraph), "Exploration graph must be a PyDiGraph"
        assert exploration_graph.is_symmetric(), "Exploration graph must have bi-directional links"

    def validate_state(self, ordered_removables: list[int]):
        tmpA = self.dtree_active_indices
        tmpB = set(self.dtree.active_nodes())
        assert tmpA == tmpB, "dtree_active_indices != active_nodes()"
        tmpC = self.untouchables | set(ordered_removables)
        if tmpB != tmpC:
            logger.warning("untouchables | ordered_removables != active_nodes()")
            logger.warning(f"  active nodes not in untouchables | ordered_removables: {tmpB - tmpC}")
            logger.warning(f"  untouchables | ordered_removables not in active nodes: {tmpC - tmpB}")

    def _generate_untouchables(self) -> set[int]:
        """Set of all terminals, fixed roots and leaf terminal parents"""
        untouchables = set(self.terminal_to_root.keys()) | set(self.terminal_to_root.values()) - {SUPER_ROOT}
        self._seed_degree1_parents(untouchables)
        return untouchables

    def _seed_degree1_parents(self, X: set[int]):
        """Add unambigous connected nodes (degree 1) to X (in-place)."""
        for node in X.copy():
            neighbors = self.index_to_neighbors[node]
            if len(neighbors) == 1:
                X.update(neighbors)
        if self._do_debug:
            logger.debug(f"Added degree 1 terminal parents to X. New  size: {len(X)}...")

    def _pyGraph_subgraph_stable(
        self, indices: list[int] | set[int], source_graph: rx.PyGraph | None = None, inclusive: bool = True
    ) -> rx.PyGraph:
        """Copies ref graph and deletes all nodes not in indices if inclusive is True.

        This generates a subgraph that has 1:1 index matching with ref_graph which eliminates
        the need for node_map. If any nodes are added to this copy those nodes will **NOT**
        have the same indices as they would in ref_graph.
        """
        dup_graph = self.ref_graph.copy() if source_graph is None else source_graph.copy()
        if not inclusive:
            dup_graph.remove_nodes_from(indices)
        else:
            dup_graph.remove_nodes_from(set(dup_graph.node_indices()) - set(indices))
        return dup_graph

    def _dump_graph_specs(self, subgraph: rx.PyGraph | None = None) -> None:
        if not self._do_debug:
            return

        # import counter for node degrees
        from collections import Counter

        if subgraph is not None:
            dump_graph = subgraph
        else:
            dump_graph = self.ref_graph

        degree_counter = Counter([dump_graph.degree(v) for v in dump_graph.node_indices()])
        basis_cycles_counts_by_length = Counter([len(cycle) for cycle in rx.cycle_basis(dump_graph)])
        component_counts_by_length = Counter([len(cc) for cc in rx.connected_components(dump_graph)])

        logger.debug(
            "Dumping graph specs...\n"
            f"Number of nodes: {dump_graph.num_nodes()}\n"
            f"Number of edges: {dump_graph.num_edges()}\n"
            f"Number of terminals: {len(self.terminal_to_root)}\n"
            f"Number of super terminals: {len([t for t in self.terminal_to_root.values() if t == SUPER_ROOT])}\n"
            f"Number of base towns: {len(self.base_towns)}\n"
            f"Number of basis cycles: {len(rx.cycle_basis(dump_graph))}\n"
            f"Basis cycles by length: {basis_cycles_counts_by_length}\n"
            f"Nodes by degree: {degree_counter}\n"
            f"Number of connected components: {rx.number_connected_components(dump_graph)}\n"
            f"Components by length: {component_counts_by_length}\n"
        )

    def solve(self) -> rx.PyDiGraph:
        """Runs the Steiner Forest approximation algorithm..."""
        # if self._do_debug:
        #     logger.debug("Running Steiner Forest approximation...")
        from api_common import get_clean_exploration_data
        import json
        # Manual testing of NodeRouter PDApproximation...
        waypoint_to_index = {v: i for i, v in enumerate(self.index_to_waypoint.values())}
        exploration_data = get_clean_exploration_data(config)
        exploration_json_dumps = json.dumps(exploration_data)
        NR = NodeRouter(exploration_json_dumps)

        start_time = time.perf_counter()
        ordered_removables = self._approximate()
        app1_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        test_result = NR.py_solve_for_terminal_pairs(list(self.testing_orig_terminals.items()))
        app2_time = time.perf_counter() - start_time
        print(f"Approximation => Python: {app1_time} Rust: {app2_time}")

        test_cost = sum(self.index_to_weight[waypoint_to_index[v]] for v in test_result)
        validation_cost = sum(self.index_to_weight[v] for v in self.dtree_active_indices)
        if test_cost != validation_cost:
            logger.error(f"Test cost != Validation cost: {test_cost} != {validation_cost}")
            print("Test result:", sorted((test_result)))
            print("Validation result:", sorted(self.dtree_active_indices))
            print(
                f"Python len active nodes is {len(self.dtree.active_nodes())} num active indices is {len(self.dtree_active_indices)} and num ordered removables is {len(ordered_removables)}"
            )
        else:
            logger.success("Python and Rust match on cost...")
        # if self._do_debug:
        #     self.validate_state(ordered_removables)

        # Processing order matters! Use both forward and reverse
        # removal combos in distinct passes for improved solutions.
        dtree_ORIG = self.dtree.py_clone()
        removables_ORIG = ordered_removables.copy()
        active_indices_ORIG = self.dtree_active_indices.copy()

        self._combo_gen_direction = False
        self._bridge_heuristics(ordered_removables)
        dtree_incumbentA = self.dtree.py_clone()
        print("----------------------------------------------")
        self.dtree = dtree_ORIG
        ordered_removables = removables_ORIG
        self.dtree_active_indices = active_indices_ORIG

        self._combo_gen_direction = True
        self._bridge_heuristics(ordered_removables)

        if dtree_incumbentA is not None:
            # Test which incumbent dominates
            incumbentA_weight = sum(self.index_to_weight[v] for v in dtree_incumbentA.active_nodes())
            incumbentA_component_count = dtree_incumbentA.num_connected_components()
            incumbent_weight = sum(self.index_to_weight[v] for v in self.dtree.active_nodes())
            incumbent_component_count = self.dtree.num_connected_components()
            if incumbentA_weight < incumbent_weight or (
                incumbentA_weight == incumbent_weight
                and incumbentA_component_count > incumbent_component_count
            ):
                self.dtree = dtree_incumbentA

        self._bridge_affected_indices = set(self.dtree.active_nodes())
        self._bridge_affected_terminals = set(self.terminal_to_root.items())
        self._bridge_affected_base_towns = self.base_towns
        assert self._terminal_pairs_connected()

        solution_graph = self._finalize(start_time)
        return solution_graph

    def _finalize(
        self,
        start_time: float,
    ) -> rx.PyDiGraph:
        """Generate a PyDiGraph with terminal set attributes from candidate.

        - re-assigns original terminal roots pairs to pairs from the approximation clusters.
        """
        subgraph = self._pyGraph_subgraph_stable(list(self.dtree.active_nodes()))
        if self._do_info:
            logger.info("Finalizing solution...")
            self._dump_graph_specs(subgraph)

        terminal_waypoints = {
            self.index_to_waypoint[t]: self.index_to_waypoint.get(r, SUPER_ROOT)
            for t, r in self.terminal_to_root.items()
        }
        all_roots_and_terminals = set(terminal_waypoints.keys()) | set(terminal_waypoints.values()) - {
            SUPER_ROOT
        }
        all_base_towns: set[int] = {self.index_to_waypoint[node] for node in self.base_towns}

        final_components: list[list[int]] = []
        terminal_pairs: dict[int, int] = {}
        terminal_sets: dict[int, list[int]] = {}
        for component in sorted(rx.connected_components(subgraph), key=len):
            waypoints = [self.index_to_waypoint[i] for i in component]

            # Extract root and terminals for this component
            if self._do_trace:
                logger.trace(f"Assigning root and terminals for component containing: {waypoints}")
            root, terminals = self._assign_root_and_terminals(
                waypoints,
                all_roots_and_terminals,  # type: ignore
                all_base_towns,
            )

            terminal_pairs.update([(t, root) for t in terminals])
            terminal_sets[root] = terminals
            final_components.append(list(component))

        if self._do_info:
            logger.info(
                f"Final solution graph contains {len(subgraph.node_indices())} nodes"
                f" in {len(final_components)} components, Total time: {time.perf_counter() - start_time:.4f}s"
            )
            logger.trace(f"Components: {final_components}")
            logger.info(f"Component terminal pairs: {terminal_pairs}")
            logger.info(f"Component terminal sets: {terminal_sets}")
            logger.info(
                f"Solution waypoints: {[subgraph[v]['waypoint_key'] for v in subgraph.node_indices()]}"
            )

        subgraph = subgraph.to_directed()
        subgraph.attrs = {
            "node_key_by_index": bidict({v: subgraph[v]["waypoint_key"] for v in subgraph.node_indices()})
        }
        set_graph_terminal_sets_attribute(subgraph, terminal_pairs)
        return subgraph

    def _assign_root_and_terminals(
        self, waypoints: list[int], roots_and_terminals: set[int], base_towns: set[int]
    ) -> tuple[int, list[int]]:
        """Create the terminal root assignments."""
        if self._do_debug:
            logger.debug(f"Assigning root and terminals for waypoints: {waypoints}")

        terminals_in_component = set(waypoints) & roots_and_terminals
        base_towns_in_component = set(waypoints) & base_towns

        # Use the first waypoint that is a base town for the cluster's root.
        # Use a basetown that is also a terminal if one is available, if not then
        # the terminals in the cluster are super-terminals and assign to the nearest
        # basetown which is assumed to be the only basetown in the tree.
        if base_town_terminals := terminals_in_component & base_towns_in_component:
            root = base_town_terminals.pop()
            terminals_in_component.remove(root)
        else:
            if len(base_towns_in_component) > 1:
                logger.warning(f"Multiple basetowns in cluster, using the first! {base_towns_in_component}")
            try:
                root = base_towns_in_component.pop()
            except KeyError:
                logger.error(f"No basetowns in cluster! {base_towns_in_component}")
                logger.error(f"Terminals in cluster: {terminals_in_component}")
                logger.error(f"waypoints: {waypoints}")
                logger.error(f"roots_and_terminals: {roots_and_terminals}")
                logger.error(f"base_towns: {base_towns}")
                raise ValueError("No basetowns in cluster!")

        return root, sorted(terminals_in_component)

    ###################################################################
    # MARK: PD Approximation
    ###################################################################

    def _approximate(self) -> list[int]:
        start_time = time.perf_counter()

        X = self._generate_untouchables()

        if self._do_debug:
            logger.debug(f"Initialized approximation with {len(X)} nodes...")
            starting_ccs = rx.connected_components(self.ref_graph.subgraph(list(X)))
            logger.debug(
                f"There are {len(starting_ccs)} connected components in X of sizes: {sorted([len(cc) for cc in starting_ccs])}",
            )

        X, iters, ordered_removables = self._primal_dual_approximation(X)

        # Initializing the dtree using the PyGraph just seems simplest...
        subgraph = self._pyGraph_subgraph_stable(X)
        for v in subgraph.node_indices():
            for u in subgraph.neighbors(v):
                self.dtree.insert_edge(u, v)
        self.dtree_active_indices = set(self.dtree.active_nodes())
        # self.validate_node_connected_components(subgraph, subgraph, "approximation initialization")

        # Ordered removables are in temporal order of going tight, sub ordered structurally
        # by sorting by waypoint key, when removing the nodes they should be processed in
        # reverse order to facilitate the removal of the latest nodes to 'go tight' first.
        # The list is reversed here and processed in forward order thoughout the remainder
        # of the algorithm and bridge heuristic processing.
        ordered_removables = self._prune_approximation(ordered_removables)
        ordered_removables = list(reversed(ordered_removables))
        # if self._do_info:
        incumbent_weight = sum(self.index_to_weight[v] for v in self.dtree.active_nodes())
        if self._do_debug:
            logger.warning(
                f"Post pruning: node count={len(self.dtree.active_nodes())}, cost={incumbent_weight}, "
                f"components={self.dtree.num_connected_components()}, "
                f"iterations={iters}, time={time.perf_counter() - start_time:.4f}s"
            )

        # remove_removables is setup to primarily handle 'bridged' components in the Bridge
        # Heuristic. To simplify the code the bridge related variables are set here to
        # cover all removables, terminals and base towns in the graph.
        self._update_bridge_affected_nodes(self.dtree_active_indices)
        freed, _freed_edges = self._remove_removables(ordered_removables)

        self.dtree_active_indices -= freed
        ordered_removables = [v for v in ordered_removables if v not in freed]

        if self._do_info:
            incumbent_weight = sum(self.index_to_weight[v] for v in self.dtree.active_nodes())
            logger.info(
                f"Initial solution: node count={len(self.dtree.active_nodes())}, cost={incumbent_weight}, "
                f"components={self.dtree.num_connected_components()}, "
                f"iterations={iters}, time={time.perf_counter() - start_time:.4f}s"
            )

        return ordered_removables

    def _primal_dual_approximation(self, X: set[int]) -> tuple[set[int], int, list[int]]:
        """Classical variant based on Demaine et al."""
        # While the main loop operations and frontier node calculations are set based the
        # violated sets identification requires subgraphing the ref_graph and running
        # connected_components on the full graph. The loop usually only iterates a half
        # dozen times.
        if self._do_info:
            logger.info("Running primal-dual approximation")

        y = [0] * self.ref_graph.num_nodes()
        ordered_removables = []
        viol_iters = 0
        while violated_sets := self._violated_sets(X):
            viol_iters += 1
            violated = set().union(*violated_sets)
            if self._do_trace:
                logger.trace(f"Violated nodes: {[self.index_to_waypoint[v] for v in violated]}")
            frontier_nodes = self._find_frontier_nodes(violated)
            tight_nodes = [v for v in frontier_nodes if y[v] == self.index_to_weight[v]]
            X.update(tight_nodes)
            ordered_removables.extend(self._sort_by_weights(tight_nodes))
            for v in frontier_nodes:
                y[v] += 1
        return X, viol_iters, ordered_removables

    def _sort_by_weights(self, numbers: list[int], weights: list[int] | None = None):
        if weights is None:
            # NOTE:
            # The sorting here was purely for deterministic testing while refactoring from networkx.
            # But amazingly enough this _dramatically_ improves results with PD!
            # Considering this is sorting by 'waypoint_key' it doesn't really represent a weight
            # that _should_ have anything to do with the component growth but since the keys
            # are somewhat geographically ordered the structural ordering of the bridge
            # and removal candidates during the bridge heuristics is altered.
            weights = [self.index_to_waypoint[i] for i in numbers]
        pairs = zip(weights, numbers)
        sorted_pairs = sorted(pairs)
        sorted_numbers = [number for _, number in sorted_pairs]
        return sorted_numbers

    def _prune_approximation(self, ordered_removables: list[int]) -> list[int]:
        """Simple straight forward pruning of non-terminal degree 1 nodes from the graph."""
        if self._do_debug:
            logger.debug("Pruning degree 1 nodes...")

        terminal_indices = set(self.terminal_to_root.keys())
        untouchable_indices = terminal_indices | self.untouchables.intersection(self.dtree_active_indices)

        while indices := [
            i for i in self.dtree_active_indices if self.dtree.degree(i) == 1 and i not in untouchable_indices
        ]:
            for i in indices:
                self.dtree.isolate_node(i)
                self.dtree_active_indices.remove(i)
            if self._do_debug:
                logger.debug(f"Removed {len(indices)} degree 1 nodes")

        ordered_removables = [v for v in ordered_removables if v in self.dtree.active_nodes()]
        return ordered_removables

    def _find_frontier_nodes(self, settlement: set[int], min_degree=0) -> set[int]:
        """Finds and returns nodes not in settlement with neighbors in settlement."""
        if self._do_debug:
            logger.debug("Finding frontier nodes...")

        frontier = set()
        for v in settlement:
            frontier.update(self.index_to_neighbors[v])
        frontier.difference_update(settlement)
        if min_degree > 0:
            frontier = {v for v in frontier if len(self.index_to_neighbors[v]) >= min_degree}

        if self._do_debug:
            logger.warning(f"Found {len(frontier)} frontier nodes for settlement of size {len(settlement)}...")

        return frontier

    def _violated_sets(self, nodes_subset: set[int]) -> list[set[int]]:
        """Returns connected components violating connectivity constraints."""
        if self._do_debug:
            logger.debug("Finding violated sets...")

        # It is fairly common in the smaller test incidents that a super terminal can
        # be placed in a Terminal Cluster Set of a component with a higher cost simply
        # because another (cheaper) component near it is non-violated. To avoid these
        # situations reset the connected pairs cache and mark all components as violated
        # until all super terminals are connected. Since it is assumed that super terminals
        # are listed last in the terminal_to_root listing use reversed to eliminate almost
        # all overhead associated with tightening and removing more frontier nodes.

        # Another scenario where a super terminal can be put in a higher cost component
        # is when it is nearer to a basetown that is a potential super-terminal root but
        # that basetown isn't a fixed root for any other terminal while the super-terminal
        # is also near enough to a higher cost component's settlement that _is_ violated,
        # meaning the violated component and super-terminal component both grow while the
        # potential super-terminal root basetown doesn't leading to the super-terminal
        # joining with the higher cost component prior to joining with the lower costing
        # basetown. A possible fix for this is to include potential roots in the violated
        # components until all super terminals are not violated where the potential roots
        # are those that are nearest to the super-terminals, and if that basetown is already
        # a fixed basetown for a terminal then nothing special needs to be done.

        has_violated_super_terminal = False
        subgraph = self._pyGraph_subgraph_stable(nodes_subset)
        connected_components = rx.connected_components(subgraph)
        violated: list[set[int]] = []
        for cc in connected_components:
            # Since the pd approximation is additive we can safely avoid duplicate checks.
            active_terminals = [
                p for p in reversed(list(self.terminal_to_root.items())) if p not in self.connected_pairs
            ]
            for terminal, root in active_terminals:
                terminal_in_cc = terminal in cc
                root_in_cc = (
                    any(b in cc for b in self.base_towns)
                    if (terminal_in_cc and root == SUPER_ROOT)
                    else root in cc
                )

                if not terminal_in_cc and not root_in_cc:
                    if root == SUPER_ROOT:
                        has_violated_super_terminal = True
                        break
                    continue
                if terminal_in_cc and root_in_cc:
                    self.connected_pairs.add((terminal, root))
                else:
                    if root == SUPER_ROOT:
                        has_violated_super_terminal = True
                        break
                    violated.append(cc)

        if has_violated_super_terminal:
            self.connected_pairs.clear()
            return connected_components
        return violated

    ###################################################################
    # MARK: Bridge Heuristics
    ###################################################################

    def _bridge_heuristics(self, ordered_removables: list[int]) -> None:
        """Bridge heuristic: find and utilize potential bridges to _increase_
        cycle counts and then identify removable articulation points that can
        improve the solution.
        """
        if not self.config.get("approximation", {}).get("post_process_bridges", False):
            logger.info("Skipping bridge heuristics")
            return

        if self._do_info:
            logger.info("Running bridge heuristics...")
        if self._do_trace:
            logger.trace(f"Initial added order: {ordered_removables}")

        bridge_start_time = time.perf_counter()
        self.bridge_start_time = bridge_start_time

        total_iter = test_pass = 0
        num_removals_checked = num_skipped_no_cycles = num_skipped_seen_before = num_skipped_no_candidates = 0

        incumbent_indices = self.dtree_active_indices.copy()

        seen_before_cache: set[frozenset[frozenset[int]]] = set()
        improved = True
        while improved:
            test_pass += 1
            improved = False

            if self._do_debug:
                logger.debug(
                    f"Starting bridge testing pass {test_pass} ({time.perf_counter() - bridge_start_time:.2f}s)"
                )
                logger.debug(
                    f"Unused graph nodes: {self.ref_graph.num_nodes() - len(incumbent_indices)}, "
                    f"incumbent subgraph nodes: {len(incumbent_indices)}"
                )
                self.validate_state(ordered_removables)

            bridges = self._bridge_generator(incumbent_indices)
            for bridge in bridges:
                total_iter += 1
                reisolate_bridge_nodes = [v for v in bridge if v not in incumbent_indices]

                # Test output to compare the Rust implementation to...
                logger.warning(
                        f"Processing bridge {bridge}... {[self.index_to_waypoint[v] for v in bridge]}"
                )
                continue

                if self._do_debug:
                    logger.debug(
                        f"Processing bridge {bridge}... {[self.index_to_waypoint[v] for v in bridge]}"
                    )
                    if self._do_debug:
                        self.validate_state(ordered_removables)
                    assert not any(v in incumbent_indices for v in bridge)

                # Produce a candidate by inserting bridge into incumbent tree
                self._connect_bridge(bridge)

                # Bridged component cycles meeting criteria can improve weight.
                if (bridge_rooted_cycles := self._bridge_rooted_cycles(bridge)) is None:
                    self.dtree.isolate_nodes(reisolate_bridge_nodes)
                    self.dtree_active_indices = incumbent_indices.copy()
                    num_skipped_no_cycles += 1
                    if self._do_debug:
                        self.validate_state(ordered_removables)
                    continue

                # Skip processing repeated bridge/cycle sets, the result will be the same.
                if self._was_seen_before(bridge, bridge_rooted_cycles, seen_before_cache):
                    self.dtree.isolate_nodes(reisolate_bridge_nodes)
                    self.dtree_active_indices = incumbent_indices.copy()
                    num_skipped_seen_before += 1
                    if self._do_debug:
                        self.validate_state(ordered_removables)
                    continue

                # Only particular nodes can be removed from a cycle
                if not (removal_candidates := self._removal_candidates(bridge, bridge_rooted_cycles)):
                    self.dtree.isolate_nodes(reisolate_bridge_nodes)
                    self.dtree_active_indices = incumbent_indices.copy()
                    num_skipped_no_candidates += 1
                    if self._do_debug:
                        self.validate_state(ordered_removables)
                    continue

                # Attempt to improve the solution by removing different combinations of removal candidates
                is_improved, removal_attempts, freed = self._improve_component(
                    bridge, removal_candidates, ordered_removables
                )
                total_iter += removal_attempts

                if is_improved:
                    incumbent_indices = set(self.dtree.active_nodes())
                    self.dtree_active_indices = incumbent_indices.copy()
                    bridges = self._bridge_generator(incumbent_indices)
                    improved = True
                    ordered_removables = [
                        v for v in ordered_removables if v not in freed
                    ] + self._sort_by_weights(list(bridge))
                    if self._do_debug:
                        self.validate_state(ordered_removables)
                    break

                # Re-isolate the bridge nodes in dtree to restore to pre-bridged state.
                self.dtree.isolate_nodes(reisolate_bridge_nodes)
                self.dtree_active_indices = incumbent_indices.copy()
                if self._do_debug:
                    self.validate_state(ordered_removables)

            if self._do_debug:
                logger.debug(
                    f"Bridge testing pass {test_pass} completed: iterations={total_iter}, "
                    f"components={self.dtree.num_connected_components()}, "
                    f"time={time.perf_counter() - bridge_start_time:.2f}s"
                )

            if not improved:
                if self._do_debug:
                    logger.debug(
                        "Bridge testing complete. No improvements found. "
                        f"removals checked= {num_removals_checked}, "
                        f"skipped no cycles= {num_skipped_no_cycles}, "
                        f"skipped seen before= {num_skipped_seen_before}, "
                        f"skipped no candidates= {num_skipped_no_candidates}, "
                        f"iterations={total_iter}, "
                        f"time={time.perf_counter() - bridge_start_time:.2f}s"
                    )
                break
        return

    def _bridge_generator(self, settlement: set[int]) -> Generator[frozenset[int], None, None]:
        """Finds bridging spans of settlement border nodes within frontier and wild frontier."""

        # ref_graph => full reference graph of all nodes and edges.
        # Contains potentially topologically constrained expansion of Settlement
        # Settlement potentially contains disjoint connected components

        # S => Settlement (induced subgraph of currently 'active' nodes of ref_graph)
        # B => Border (nodes in Settlement with ref_graph neighbors in Frontier)
        # F => Frontier (non-settled nodes with ref_graph neighbors in Settlement)
        # W => Wild Frontier (non-settled nodes with no settled ref_graph neighbors)

        # 𝓕 => Fringe edges (edges in ref_graph connecting border and frontier nodes)
        # 𝓑 => bridging span of nodes not in S that connect distinct nodes in S.

        # Let ring 0 => B (in most cases B == S since S is most commonly a Steiner Forest/Tree)
        # Let ring 1 => F0 be an eccentric ring around {B}
        # Let ring 2 => F1 be an eccentric ring around {F0|B}
        # Let ring 3 => F2 be an eccentric ring around {F1|F0|B}
        # ...

        max_frontier_rings = 3  # number rings around settlement
        ring_combo_cutoff = [0, 3, 2, 2]  # max width of connected run _along_ outermost ring.

        rings: list[set[int]] = []
        seen_candidate_pairs: set[tuple[int, int]] = set()
        yielded: set[frozenset[int]] = set()

        def descend_to_yield_bridges(
            ring_idx: int, current_nodes: set[int] | frozenset[int], bridge: frozenset[int]
        ) -> Generator[frozenset[int], None, None]:
            """Descend from current ring to settlement frontier (F0), yielding bridges connecting ≥2 S nodes."""
            nonlocal seen_candidate_pairs
            assert ring_idx > 0

            # Base case (Settlement Frontier): yield and return
            if ring_idx == 1:
                if bridge not in yielded:
                    s_neighbors = set().union(*(self.index_to_neighbors[v] for v in current_nodes)) & rings[0]
                    if len(s_neighbors) >= 2:
                        yielded.add(bridge)
                        yield bridge
                    else:
                        logger.warning("  too few neighbors: skipping...")
                else:
                    logger.warning(f"  previously yielded:skipping...")
                return

            # Collect and process pairwise combinations of current_node candidates...
            # Candidates are current_nodes neighbors in ring_idx - 1
            logger.warning(f"  *descending to yield with current_nodes={current_nodes} => {[self.index_to_waypoint[v] for v in current_nodes]}")
            print("num seen candidate pairs upon entry: ", len(seen_candidate_pairs))
            inner_ring = rings[ring_idx - 1]
            candidates = list(
                set.union(*(self.index_to_neighbors[n] for n in current_nodes)) & inner_ring
            )
            candidates = sorted(list(candidates))

            for i in range(len(candidates) - 1):
                u = candidates[i]
                for j in range(i + 1, len(candidates)):
                    v = candidates[j]
                    # This filter works for all depths, which is fine for a small planar graph with
                    # limited max ring depth. It may be too restrictive with higher max ring depth.
                    candidate_pair = (u, v) if u < v else (v, u)
                    print("    checking candidate pair: ", candidate_pair)
                    if candidate_pair not in seen_candidate_pairs:
                        if candidate_pair == (774, 775):
                            print("*********** Inserting candidate pair (774, 775)")
                        seen_candidate_pairs.add(candidate_pair)
                        yield from descend_to_yield_bridges(ring_idx - 1, {u, v}, bridge | frozenset({u, v}))
                    else:
                        logger.warning("  candidate pair seen before: skipping...")


        # Populate ring0 (Settlement border)
        rings.append(settlement.copy())
        seen_nodes = settlement.copy()
        # logger.warning(f"{settlement=} => {[self.index_to_waypoint[v] for v in settlement]}")

        while len(rings) <= max_frontier_rings:
            # Populate the new outermost ring
            nodes = self._find_frontier_nodes(seen_nodes, min_degree=2)
            logger.warning(f"  {nodes=} \n=> {[self.index_to_waypoint[v] for v in nodes]}")
            seen_nodes |= nodes
            ring_idx = len(rings)
            nodes = sorted(list(nodes))
            rings.append(set(nodes))

            # Phase 1:
            # Yield single node bridges from outermost ring connecting with >=2 neighbors in inner ring.
            # NOTE: Validating inner ring neighbors does not strictly need to be done but reduces workload.
            inner_ring = rings[ring_idx - 1]
            for node in nodes:
                neighbors = self.index_to_neighbors[node] & inner_ring
                if len(neighbors) < 2:
                    continue

                first, *rest = neighbors
                inner_ring_neighbors = self.index_to_neighbors[first] & inner_ring
                if not any(inner_ring_neighbors & self.index_to_neighbors[n] for n in rest):
                    yield from descend_to_yield_bridges(ring_idx, {node}, frozenset({node}))

            # Phase 2:
            # Yield multi-node bridges from outermost ring.

            # NOTE: Each ring is only a single node 'thick' so the `all_pairs_all_simple_paths`
            # is an efficient way to obtain all size constrained connected runs within the ring
            # along with the associated endpoint nodes.
            subgraph = self._pyGraph_subgraph_stable(nodes)
            node_indices = sorted(list(nodes))
            # tmp = rx.all_pairs_dijkstra_shortest_paths(subgraph, lambda x: 1.0)
            seen_endpoints: set[tuple[int, int]] = set()

            # Connected runs within the current outer ring make up the multi-node bridges for the ring.
            # They exist as connected runs between u and v.

            # NOTE: These runs can be accumulated and made available to the descend function to increase
            # the spread along the inner rings while descending but since each ring consists only of
            # F nodes each node in the span would have corresponding nodes in the inner ring and the
            # span would need to have a lower weight to improve the incumbent solution which wouldn't
            # happen based on the PD approximation.
            for i in range(len(nodes) - 1):
                u = node_indices[i]
                for j in range(i + 1, len(nodes)):
                    v = node_indices[j]

                    key = (u, v) if u < v else (v, u)
                    if key in seen_endpoints:
                        continue
                    seen_endpoints.add(key)

                    if self.index_to_neighbors[u] & self.index_to_neighbors[v] & inner_ring:
                        continue

                    path = rx.dijkstra_shortest_paths(subgraph, u, v, lambda x: 1.0)
                    if not path or len(path[v]) > ring_combo_cutoff[ring_idx]:
                        continue

                    print(f"  path: {path[v]}")
                    combo = frozenset(path[v])
                    logger.warning(f"  phase2 descending on {u} {v} {combo=} => {self.index_to_waypoint[u]} {self.index_to_waypoint[v]} {[self.index_to_waypoint[i] for i in combo]}")
                    yield from descend_to_yield_bridges(ring_idx, combo, combo)


    def _connect_bridge(self, bridge: set[int] | frozenset[int]) -> None:
        """Applies bridge to dtree."""
        # Insert edges connecting bridge nodes to their active neighbors.
        # Use tmp to store the whole bridge, deplete by moving from tmp to dtree
        # when the node in tmp has an active neighbor in dtree.
        tmp = set(bridge.copy())
        moved_node = True
        while tmp and moved_node:
            moved_node = False
            for v in tmp.copy():
                active_neighbors = self.index_to_neighbors[v] & self.dtree_active_indices
                for u in active_neighbors:
                    if self.dtree.insert_edge(v, u) == -1:
                        logger.error(f"Edge insertion failed for {(v, u)} in bridge {bridge}")
                if active_neighbors:
                    self.dtree_active_indices.add(v)
                    tmp.remove(v)
                    moved_node = True
                    if self._do_trace:
                        logger.trace(
                            f"active_neighbors for v={v} ({self.index_to_waypoint[v]}): {active_neighbors} ({[self.index_to_waypoint[i] for i in active_neighbors]})"
                        )
            if not moved_node:
                logger.warning(f"Failed to move bridge nodes: {tmp} from bridge {bridge} to dtree")

        if self._do_debug:
            component = self.dtree.node_connected_component(list(bridge)[0])
            logger.debug(
                f"Bridged component of len {len(component)} generated from bridge {bridge}... "
                f"({[self.index_to_waypoint[v] for v in bridge]})"
            )

    def _bridge_rooted_cycles(self, bridge: set[int] | frozenset[int]) -> list[frozenset[int]] | None:
        """Isolate and filter cycle basis rooted at a bridge node."""
        if self._do_debug:
            logger.debug(f"Processing bridge {bridge}... ({[self.index_to_waypoint[v] for v in bridge]})")

        # logger.warning("dtree Potential cycles:")
        # for cycle in self.dtree.cycle_basis(root=list(bridge)[0]):
        #     logger.warning(f"  {cycle}, {[self.index_to_waypoint[v] for v in cycle]}")
        # logger.warning("pygraph Potential cycles:")
        # tmp = self._pyGraph_subgraph_stable(self.dtree_active_indices)
        # for cycle in rx.cycle_basis(tmp, root=list(bridge)[0]):
        #     logger.warning(f"  {cycle}, {[self.index_to_waypoint[v] for v in cycle]}")

        cycles = [
            frozenset(cycle)
            for cycle in self.dtree.cycle_basis(root=list(bridge)[0])
            if len(cycle) >= (2 + len(bridge)) and any(i in cycle for i in bridge)
        ]

        if self._do_trace:
            logger.trace(f"Bridge {[self.index_to_waypoint[v] for v in bridge]}: {len(cycles)} cycles")
            if not cycles:
                logger.trace(f"Skipping bridge {[self.index_to_waypoint[v] for v in bridge]}: no cycles")
            else:
                logger.trace(
                    f"Created cycles: {[[self.index_to_waypoint[v] for v in cycle] for cycle in cycles]}"
                )

        if not cycles:
            return None
        return cycles

    def _was_seen_before(
        self,
        bridge: set[int] | frozenset[int],
        cycles: list[frozenset[int]],
        seen_before: set[frozenset[frozenset[int]]],
    ) -> bool:
        this_set = frozenset([frozenset(bridge)] + cycles)
        if this_set in seen_before:
            if self._do_debug:
                logger.debug(
                    f"Skipping {'node' if len(bridge) == 1 else 'edge'} bridge {bridge}, seen before."
                )
            return True
        seen_before.add(this_set)
        return False

    def _removal_candidates(
        self, bridge: set[int] | frozenset[int], cycles: list[frozenset[int]]
    ) -> list[tuple[int, int]]:
        if self._do_debug:
            logger.debug(
                f"Finding removal candidates for bridge={bridge}... {[self.index_to_waypoint[v] for v in bridge]}"
            )

        threshold = len(cycles) + 1
        candidates = set().union(*cycles) - self.untouchables - bridge

        # # MARK: TESTBLOCK
        # if bridge == frozenset({611, 612, 654, 656, 657, 659, 660, 662, 665, 604, 605, 606, 607}):
        #     print("dbg")

        #     # Remove candidates one-by-one and validate connectivity
        #     # Regardless of connectivity restore the node
        #     def __terminal_connected(terminal, root):
        #         return (
        #             self.dtree.query(terminal, root)
        #             if root != SUPER_ROOT
        #             else any(self.dtree.query(terminal, b) for b in self.base_towns)
        #         )

        #     def __terminals_pairs_connected():
        #         for terminal, root in self.terminal_to_root.items():
        #             if not __terminal_connected(terminal, root):
        #                 return False
        #         return True

        #     validated_candidates = []
        #     for v in candidates:
        #         deleted_edges = []
        #         for u in self.index_to_neighbors[v] & self.dtree_active_indices:
        #             if self.dtree.delete_edge(v, u) != -1:
        #                 deleted_edges.append((v, u))
        #         if __terminals_pairs_connected():
        #             validated_candidates.append(v)
        #         for v, u in deleted_edges:
        #             self.dtree.insert_edge(v, u)

        #     dtree_candidates: list[tuple[int, int]] = [(v, self.index_to_weight[v]) for v in validated_candidates]
        #     print("dbg")
        # # MARK: TESTCLOCK_END

        dtree_candidates: list[tuple[int, int]] = [
            (v, self.index_to_weight[v]) for v in candidates if self.dtree.degree(v) <= threshold
        ]

        if self._do_trace:
            logger.trace(
                f"Cycles= {len(cycles)}, "
                f"Candidates= {len(candidates)}, "
                f"Degree threshold= {threshold},\n"
                f"candidate indices= {[v for v, _ in dtree_candidates]}\n"
                f"Candidate nodes= {[self.index_to_waypoint[v] for v, _ in dtree_candidates]}"
            )

        return dtree_candidates

    def _improve_component(
        self,
        bridge: set[int] | frozenset[int],
        removal_candidates: list[tuple[int, int]],
        ordered_removables: list[int],
    ) -> tuple[bool, int, set[int] | frozenset[int]]:
        """Test the removal of filtered combinations of removal candidates."""
        if self._do_debug:
            logger.debug(
                f"Improving component for bridge ({[self.index_to_waypoint[v] for v in bridge]}) using removal candidates ({[self.index_to_waypoint[v] for v, _ in removal_candidates]})..."
            )
        if self._do_trace:
            logger.trace(f"bridge={bridge}, removal_candidates={removal_candidates}")

        # NOTE: Each node removal alters the connected components of the graph.

        removal_attempts = 0
        max_removal_attempts = self.max_removal_attempts

        bridged_component = set(self.dtree.node_connected_component(list(bridge)[0]))
        self._update_bridge_affected_nodes(bridged_component)

        bridge_weight = sum(self.index_to_weight[v] for v in bridge)
        incumbent_weight = sum(self.index_to_weight[v] for v in self.dtree_active_indices) - bridge_weight
        incumbent_component_count = self.dtree.num_connected_components()

        freed = set()
        removal_set = set()

        combo_gen = self._weighted_range_combo_generator(removal_candidates, bridge_weight, len(bridge))
        while (removal_set := next(combo_gen, None)) is not None:
            if removal_attempts == max_removal_attempts:
                logger.debug(f"Breaking bridge check at {bridge} after {removal_attempts} attempts")
                break
            removal_attempts += 1

            if self._do_trace:
                logger.trace(
                    f"Removing removal set indices {removal_set}... {[self.index_to_waypoint[rc] for rc in removal_set]}"
                )

            # Mutate the bridged component by isolating removal set node(s)...
            deleted_edges = []
            for v in removal_set:
                neighbors = self.index_to_neighbors[v] & self._bridge_affected_indices
                for u in neighbors:
                    self.dtree.delete_edge(v, u)
                    deleted_edges.append((v, u))
                    if self._do_trace:
                        logger.trace(
                            f"improve_component: removed dtree edges: {deleted_edges} -> {[(self.index_to_waypoint[v], self.index_to_waypoint[u]) for v, u in deleted_edges]}"
                        )

            # Connectivity testing for the removal_set.
            if not self._terminal_pairs_connected():
                # Connectivity broke: restore the removal set nodes to previous state.
                for v, u in deleted_edges:
                    self.dtree.insert_edge(v, u)
                    if self._do_trace:
                        logger.trace(
                            f"improve_component: restored dtree edges: {deleted_edges} -> {[(self.index_to_waypoint[v], self.index_to_waypoint[u]) for v, u in deleted_edges]}"
                        )
                if self._do_debug:
                    logger.debug(f"Skipping removal set {removal_set}... not connected")
                continue

            active_component_indices = bridged_component - set(removal_set)
            self._update_bridge_affected_nodes(active_component_indices)
            ordered_removables = [v for v in ordered_removables if v not in removal_set]

            # Attempt to remove additional nodes using the ordered set.
            freed, freed_edges = self._remove_removables(ordered_removables)  # TESTING
            # freed, freed_edges = self._remove_removables(list(bridge) + ordered_removables)  # TESTING
            active_component_indices -= freed
            new_weight = sum(self.index_to_weight[v] for v in self.dtree.active_nodes())

            # Test if mutated dominates incumbent.
            # mutated started as incumbent (a single component) and may have split a terminal
            # set cluster while retaining the same (or improving) the cost.
            if incumbent_weight < new_weight or (
                incumbent_weight == new_weight
                and self.dtree.num_connected_components() == incumbent_component_count
            ):
                # We need to restore the removal_set __and__ freed edges...
                for v, u in deleted_edges + freed_edges:
                    self.dtree.insert_edge(v, u)
                    if self._do_trace:
                        logger.trace(
                            f"improve_component: restored dtree edges: {deleted_edges} -> {[(self.index_to_waypoint[v], self.index_to_waypoint[u]) for v, u in deleted_edges]}"
                        )
                if self._do_debug:
                    logger.debug(f"Skipping removal set {removal_set}... failed to dominate!")
                continue

            if self._do_info:
                node_count = len(self.dtree.active_nodes())
                new_cc_count = self.dtree.num_connected_components()
                logger.success(
                    f"Improved component: node count={node_count}, cost={new_weight}, components={new_cc_count}, "
                    f"iterations={removal_attempts} "
                    f"bridge={[self.index_to_waypoint[v] for v in bridge]}, "
                    f"bridge time={time.perf_counter() - self.bridge_start_time:.4f}s"
                )
            if self._do_trace:
                logger.trace(f"Solution improved using removal set {removal_set} and freed {freed}...")

            return True, removal_attempts, freed | set(removal_set)

        return False, removal_attempts, set()

    def _weighted_range_combo_generator(
        self, items: list[tuple[int, int]], bridge_cost: int, bridge_nodes: int
    ) -> Generator[list[int], None, None]:
        """Generates combinations whose total weight is in [bridge_cost, bridge_nodes * max_node_weight].
        Yields:
            combination: list of node indices.
        """
        # NOTE: Changing of removal candidate generation alters outcomes...
        # - The difference is in the order of removal_set removals
        # - Results are all within +/- 1 on cost and #cc
        # - Example difference of +1 cost on workerman incident 265
        # - Example difference of -1 cost with +1 cc on workerman incident 410
        # items.sort(key=lambda x: x[1], reverse=False)  # sorts by weight - better for well established small terminal set clusters
        # items.sort(key=lambda x: x[1], reverse=True)  # sorts by weight - better for large intermixed terminal set clusters

        items.sort(key=lambda x: x[1], reverse=self._combo_gen_direction)

        max_removal_weight = bridge_nodes * self.max_node_weight

        def backtrack(index, current_combination, current_weight, target_weight, total_available_weight):
            if current_weight >= target_weight:
                yield [item[0] for item in current_combination]
                return
            if index == len(items):
                return
            if current_weight + total_available_weight < target_weight:
                return
            # Include
            yield from backtrack(
                index + 1,
                current_combination + [items[index]],
                current_weight + items[index][1],
                target_weight,
                total_available_weight - items[index][1],
            )
            # Exclude
            yield from backtrack(
                index + 1,
                current_combination,
                current_weight,
                target_weight,
                total_available_weight,
            )

        total_available_weight = sum(item[1] for item in items)
        for target_weight in range(bridge_cost, max_removal_weight + 1):
            yield from backtrack(0, [], 0, target_weight, total_available_weight)

    def _update_bridge_affected_nodes(self, affected_component: set[int]) -> None:
        """Updates self._bridge_* variables with relevant bridged component nodes."""
        if self._do_debug:
            logger.debug("Updating bridge-affected terminals and unaffected components...")

        self._bridge_affected_indices = set(affected_component)
        self._bridge_affected_terminals = {
            (t, r) for t, r in self.terminal_to_root.items() if t in self._bridge_affected_indices
        }
        self._bridge_affected_base_towns = self.base_towns & self._bridge_affected_indices

        if self._do_trace:
            logger.trace(
                f"Affected terminals: {[self.index_to_waypoint[t] for t, _ in self._bridge_affected_terminals]}"
            )

    #####
    # MARK: Connectivity Testing
    #####

    def _remove_removables(self, ordered_removables: list[int]) -> tuple[set[int], list[tuple[int, int]]]:
        """Attempt removals of each node in ordered_removables.

        NOTE: This function should only be entered after terminal pairs connected check succeeds.
        """
        if self._do_debug:
            logger.debug(f"Performing removals on {len(ordered_removables)} added nodes...")

        freed = set()
        freed_edges = []

        for v in ordered_removables:
            if v not in self._bridge_affected_indices:
                continue
            if self._do_trace:
                logger.trace(
                    f"Processing node {v} ({self.index_to_waypoint[v]})"
                    f" with active neighbors {list(self.dtree.neighbors(v))}"
                )

            # Simulate removal by isolating the node
            deleted_edges = []
            for u in self.index_to_neighbors[v] & self._bridge_affected_indices:
                if self.dtree.delete_edge(v, u) != -1:
                    deleted_edges.append((v, u))
                if self._do_trace:
                    logger.trace(f"Removed dtree edges: {deleted_edges}")

            if self._terminal_pairs_connected():
                # Finalize removal
                self._bridge_affected_indices.remove(v)
                self._bridge_affected_base_towns.discard(v)
                freed.add(v)
                freed_edges.extend(deleted_edges)
                if self._do_trace:
                    logger.trace(f"Removed node {v=} ({self.index_to_waypoint[v]})...")
            else:
                # Restore broken connectivity
                for v, u in deleted_edges:
                    self.dtree.insert_edge(v, u)
                if self._do_trace:
                    logger.trace(f"Restored node {v} ({self.index_to_waypoint[v]}) incident edges...")

        if self._do_debug:
            logger.trace(f"freed nodes: {[self.index_to_waypoint[v] for v in freed]}")

        return freed, freed_edges

    def _terminal_pairs_connected(self) -> bool:
        """Checks if affected terminal pairs are connected in the subgraph."""
        if self._do_trace:
            logger.trace(
                f"Checking {len(self._bridge_affected_terminals)} affected terminals on graph with "
                f"{len(self.dtree.active_nodes())} active nodes..."
            )

        all_connected = True
        for terminal, root in self._bridge_affected_terminals:
            all_connected = self.terminal_is_connected(terminal, root)
            if not all_connected:
                break

        if self._do_trace:
            if all_connected:
                logger.trace(f"All {len(self._bridge_affected_terminals)} affected terminals connected")
            else:
                logger.trace("Not all affected terminals connected!")
        return all_connected

    def terminal_is_connected(self, terminal: int, root: int) -> bool:
        return (
            self.dtree.query(terminal, root)
            if root != SUPER_ROOT
            else any(self.dtree.query(terminal, b) for b in self._bridge_affected_base_towns)
        )


#####
# MARK: Main entry
#####


def optimize_with_terminals(
    exploration_graph: rx.PyDiGraph, terminals: dict[int, int], config: dict
) -> ResultDict:
    """Public-facing function to optimize graph with terminals."""
    logger.debug(f"Optimizing graph with {len(terminals)} terminals...")
    solver = PrimalDualNWSF(exploration_graph.copy(), terminals, config)
    solution_graph = solver.solve()
    objective_value = sum(v["need_exploration_point"] for v in solution_graph.nodes())
    return ResultDict({"solution_graph": solution_graph, "objective_value": objective_value})


if __name__ == "__main__":
    import testing as test

    config = ds.get_config("/home/thell/nwsf_rust/python/nwsf_rust/pd_approximation.toml")
    set_logger(config)

    if config.get("actions", {}).get("baseline_tests", False):
        success = test.baselines(optimize_with_terminals, config)
        if not success:
            raise ValueError("Baseline tests failed!")
        logger.success("Baseline tests passed!")

    if config.get("actions", {}).get("scaling_tests", False):
        total_time_start = time.perf_counter()
        # for budget in [415]:
        # for budget in range(5, 555, 5):
        for budget in [100]:
            print("\n" + "*" * 100)
            print(f"Test: optimal terminals for budget of {budget}")
            test.workerman_terminals(optimize_with_terminals, config, budget, False)
            print("-" * 72)
            # test.workerman_terminals(optimize_with_terminals, config, budget, True)
        # for percent in [9, 100]:
        # for percent in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        #     print("\n" + "*" * 100)
        #     print(f"Test: random terminals representing {percent} percent coverage:")
        #     test.random_terminals(optimize_with_terminals, config, percent, False, max_danger=5)
        #     print("-" * 72)
        #     test.random_terminals(optimize_with_terminals, config, percent, True, max_danger=5)
        print(f"Cumulative testing runtime: {time.perf_counter() - total_time_start:.2f}s")

    # # fmt:off
    # tst_set = {1826: 1795, 1827: 1795, 1828: 1795, 1829: 1795, 1879: 1857, 1880: 1857, 1882: 1857, 132: 1, 176: 1, 175: 1, 488: 301, 905: 608, 1821: 1781, 1822: 1781, 1830: 1781, 1886: 1857, 1881: 1853, 1884: 1853, 1885: 1853, 1889: 1857, 1893: 1858, 1894: 1853, 464: 301, 1820: 1781, 1890: 1853, 1891: 1858, 1892: 1858, 1895: 1853, 1896: 1853, 1815: 1781, 1813: 1795, 136: 1, 160: 1, 439: 301, 440: 301, 1513: 1319, 1527: 1319, 1538: 1301, 1537: 1301, 1904: 1834, 1899: 1834, 1883: 1853, 1897: 1853, 1535: 1301, 1565: 1301, 1887: 1853, 1888: 1853, 1913: 602, 1912: 602, 1209: 1101, 1206: 1141, 1510: 1319, 1514: 1314, 1555: 1380, 1642: 1623, 1645: 1623, 1909: 1843, 852: 601, 853: 601, 842: 601, 840: 601, 908: 608, 1515: 1314, 1902: 1843, 1903: 1843, 1907: 1843, 1906: 1843, 443: 301, 1522: 1319, 1523: 1319, 455: 301, 476: 301, 480: 301, 1521: 1319, 952: 602, 1778: 1750, 184: 1, 183: 1, 144: 1, 1911: 602, 1208: 1101, 1561: 1301, 1562: 1301, 1558: 1301, 1556: 1301, 1516: 1314, 1636: 1623, 1637: 1623, 1771: 1750, 1772: 1750, 1517: 1314, 903: 601, 914: 601, 1687: 1649, 913: 601, 912: 601, 1685: 301, 1686: 301, 1713: 1691, 1681: 1649, 1688: 1649, 1220: 301, 1201: 301, 1213: 301, 1219: 301, 1212: 301, 1216: 301, 901: 601, 902: 601, 907: 608, 1683: 302, 1684: 302}
    # exploration_graph = get_exploration_graph(config)
    # result = optimize_with_terminals(exploration_graph, tst_set, config)
    # result_graph = result["solution_graph"]
    # cost = sum([n["need_exploration_point"] for n in result_graph.nodes()])
    # print(f"node graph cost: {cost}")

    # workerman_plantzones = [n["need_exploration_point"] for n in result_graph.nodes() if n["is_workerman_plantzone"]]
    # print(f"There are {len(workerman_plantzones)} plantzones with cost: {sum(workerman_plantzones)}")

    # nonworkerman_plantzones = [n["need_exploration_point"] for n in result_graph.nodes() if not n["is_workerman_plantzone"] and not n["is_base_town"]]
    # print(f"There are {len(nonworkerman_plantzones)} non-plantzone/non-base-twon nodes with cost: {sum(nonworkerman_plantzones)}")
