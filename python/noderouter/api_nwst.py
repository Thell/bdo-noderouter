# api_nwst.py

import sys
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import time

import rustworkx as rx
from loguru import logger
from rustworkx import PyDiGraph
from sfpgre_cch import get_interactivity_edges

from api_common import PAYLOAD_WEIGHT_KEY
from api_nwst_types import (
    BlockCosts,
    BlockedInteractionEdges,
    BlockKey,
    BlockMask,
    BlockResults,
    BlockTask,
    CompositeSolution,
    ConnectedComponentMappingKey,
    ConnectedComponentMappings,
    Cost,
    CoverageBits,
    CoverageRepresentatives,
    CoverageSets,
    MaskedBlockCosts,
    MaskedBlockSolutionMasks,
    NodeIndex,
    RootIndex,
    SolutionMask,
    SuperRootIndex,
    TerminalsList,
)
from api_nwstp_problem import TreeProblem
from api_nwstp_solver import solve_tree
from api_rx_pydigraph import subgraph_stable

# Yes, really, empirically it is 42, lol.
TREE_PROBLEM_TASK_SEQUENTIAL_THRESHOLD = 42

TREE_PROBLEM_MAX_WORKERS = 14
TREE_PROBLEM_MAX_CHUNKSIZE = 256
TREE_PROBLEM_TARGET_CHUNKS = 16
TREE_PROBLEM_PROGRESS_INTERVAL = 1_000


def _connected_component_mappings(
    G: PyDiGraph,
    coverage_representatives: CoverageRepresentatives,
    super_root_index: SuperRootIndex | None = None,
) -> tuple[
    dict[NodeIndex, ConnectedComponentMappingKey],
    dict[ConnectedComponentMappingKey, ConnectedComponentMappings],
]:
    """Generates a set of mappings for each connected component of the graph for use in TreeProblems.

    NOTE: The presence of the super root would make for a single connected component and is removed
          via switching from WCC to SCC when sr_index is not None.

    NOTE: Since scipstp has issues with graphs containing multiple components, we need to partition the graph
          into connected components and handle the adj mapping and weights for each component separately.
    """
    logger.trace("    generating connected component mappings...")

    component_data = {}
    representatives_component_map = {}

    # NOTE: we can't always use weakly connected components because the super root may be present
    # which would make a (worthless) single component.
    # However SCC is notably slower than WCC so we default to WCC unless the super root is present
    cc_fn = (
        rx.strongly_connected_components
        if super_root_index is not None and G.has_node(super_root_index)
        else rx.weakly_connected_components
    )
    components = cc_fn(G)
    for cc_i, cc in enumerate(components):
        # rustworkx wcc returns as sets but scc returns as lists
        cc = set(cc)

        # Representative reachability for component generation...
        # NOTE: Super root is always isolated in a SCC, so we can ignore it...
        cc_reachable = cc & coverage_representatives
        if not cc_reachable:
            continue

        for r in cc_reachable:
            representatives_component_map[r] = cc_i

        # Even though scipstp NWSTP is fully a node weighted setup the rustworkx graph isn't pickleable
        # so we need to pass in the adjacency map and node weights
        adj_map = {}
        for u in cc:
            # NOTE: We only use predecessors; which is constistent with the graph reduction engine's
            #       `reduction_neighbors` method (see its' docstring).
            adj_map[u] = sorted(set(G.predecessor_indices(u)))

        # For index masking...
        nodes_list = list(G.node_indices())
        node_index = {u: i for i, u in enumerate(nodes_list)}

        # DIMACS node ids are 1-based, contiguous
        dimacs_id = {u: i + 1 for i, u in enumerate(nodes_list)}
        inv_dimacs_id = {i + 1: u for i, u in enumerate(nodes_list)}

        component_data[cc_i] = {
            "component": cc,
            "reachable": cc_reachable,
            "adj_map": adj_map,
            "nodes_list": nodes_list,
            "node_index_map": node_index,
            "dimacs_id_map": dimacs_id,
            "inv_dimacs_id_map": inv_dimacs_id,
        }

    logger.trace(
        f"      identified {len(component_data)} reachability component(s): {[len(cc_data.get('reachable')) for cc_data in component_data.values()]}"
    )

    return representatives_component_map, component_data


def _interactivity_graph(
    paths_G: PyDiGraph,
    interactivity_G: PyDiGraph,
    coverage_sets: CoverageSets,
    super_root_index: SuperRootIndex | None = None,
    interactivity_edges: list[tuple[RootIndex, RootIndex, float]] | None = None,
) -> tuple[PyDiGraph, BlockedInteractionEdges]:
    """Creates the interactivity graph for the given reduced state graph over the given coverage sets.

    If interactivity_edges is provided, it is used as-is instead of being computed here --
    this lets a caller apply its own edge-level filtering (e.g. the super pathway's
    _filter_blocked_st_interactivity_edges) on the raw edge list before the graph is built
    and Steiner-filtered, so both filters run against the same edge set in the right order.
    """
    coverage_representatives = set(coverage_sets.keys())

    if interactivity_edges is None:
        node_weight_map = {u: paths_G[u][PAYLOAD_WEIGHT_KEY] for u in paths_G.node_indices()}

        # NOTE: We have to use self.graph to capture st -> sr paths here...
        # Union sets of all nodes for all shortest paths for each t -> r for each t in terminal_set r
        arbor_rt_all_shortest_path_unions = {r: set() for r in coverage_representatives}
        for r, terminals in coverage_sets.items():
            if r not in coverage_representatives:
                continue
            for t in terminals:
                paths = rx.all_shortest_paths(paths_G, t, r, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
                arbor_rt_all_shortest_path_unions[r].update(*paths)

        # NOTE: We can't use self.graph with a super root or gaps will collapse
        interactivity_edges = get_interactivity_edges(
            interactivity_G, coverage_sets, node_weight_map, arbor_rt_all_shortest_path_unions
        )

    interactivity_graph = subgraph_stable(paths_G, coverage_representatives)
    interactivity_graph.remove_edges_from(interactivity_graph.edge_list())
    interactivity_graph.add_edges_from(interactivity_edges)

    connected_pairs = _steiner_connected_root_pairs(paths_G, coverage_sets, super_root_index)
    blocked_edges = _blocked_interactivity_edges(interactivity_graph, connected_pairs)

    return interactivity_graph, blocked_edges


def _steiner_connected_root_pairs(
    G: PyDiGraph, coverage_sets: CoverageSets, super_root_index: SuperRootIndex | None = None
) -> BlockedInteractionEdges:
    """Root pairs (r, s) that share a genuine Steiner-only path in G.

    A composite solve of r and s only has synergy to exploit if some path
    connecting their arbors is not underwritten by a third root's already-
    committed demand nodes -- that node's cost is paid in full by whichever
    block contains it regardless of partitioning, so a detour through it
    buys no sharing. Certifies exclusion only: absence of any Steiner-only
    path means the gap/drt admission test was measuring proximity through
    someone else's territory, not real interactivity.
    """
    non_steiner = set(coverage_sets.keys()) | set().union(*coverage_sets.values())
    steiner_nodes = set(G.node_indices()) - non_steiner

    if super_root_index is not None:
        non_steiner.discard(super_root_index)
        steiner_nodes.discard(super_root_index)

    steiner_sub = subgraph_stable(G, steiner_nodes)
    steiner_components = rx.weakly_connected_components(steiner_sub)

    node_to_comp: dict[int, int] = {}
    for comp_i, comp in enumerate(steiner_components):
        for n in comp:
            node_to_comp[n] = comp_i

    root_components: dict[int, set[int]] = {}
    for r, terminals in coverage_sets.items():
        comps = set()
        for d in terminals | {r}:
            for nbr in G.predecessor_indices(d):
                if nbr in node_to_comp:
                    comps.add(node_to_comp[nbr])
        root_components[r] = comps

    connected_pairs = set()
    roots = sorted(root_components.keys())
    for i, r in enumerate(roots):
        for s in roots[i + 1 :]:
            if root_components[r] & root_components[s]:
                connected_pairs.add((r, s))

    return connected_pairs


def _steiner_connected_root_components(
    G: PyDiGraph, coverage_sets: CoverageSets, super_root_index: SuperRootIndex | None = None
) -> dict[int, set[int]]:
    """Root pairs (r, s) that share a genuine Steiner-only path in G.

    A composite solve of r and s only has synergy to exploit if some path
    connecting their arbors is not underwritten by a third root's already-
    committed demand nodes -- that node's cost is paid in full by whichever
    block contains it regardless of partitioning, so a detour through it
    buys no sharing. Certifies exclusion only: absence of any Steiner-only
    path means the gap/drt admission test was measuring proximity through
    someone else's territory, not real interactivity.
    """
    non_steiner = set(coverage_sets.keys()) | set().union(*coverage_sets.values())
    steiner_nodes = set(G.node_indices()) - non_steiner

    if super_root_index is not None:
        non_steiner.discard(super_root_index)
        steiner_nodes.discard(super_root_index)

    steiner_sub = subgraph_stable(G, steiner_nodes)
    steiner_components = rx.weakly_connected_components(steiner_sub)

    node_to_comp: dict[int, int] = {}
    for comp_i, comp in enumerate(steiner_components):
        for n in comp:
            node_to_comp[n] = comp_i

    root_components: dict[int, set[int]] = {}
    for r, terminals in coverage_sets.items():
        comps = set()
        for d in terminals | {r}:
            for nbr in G.predecessor_indices(d):
                if nbr in node_to_comp:
                    comps.add(node_to_comp[nbr])
        root_components[r] = comps

    return root_components


def _blocked_interactivity_edges(
    interactivity_graph: PyDiGraph, connected_pairs: set[tuple[NodeIndex, NodeIndex]]
) -> set[tuple[NodeIndex, NodeIndex]]:
    blocked_edges = set()
    for u, v in list(interactivity_graph.edge_list()):
        key = (u, v) if u < v else (v, u)
        if key not in connected_pairs:
            blocked_edges.add((u, v))
    return blocked_edges


def _problem_generator(
    instance_id: str,
    component_data: dict[ConnectedComponentMappingKey, ConnectedComponentMappings],
    node_weight_map: dict[NodeIndex, Cost],
    tasks: list[BlockTask],
    do_debug: bool = False,
    mip_validation: bool = True,
) -> Generator[TreeProblem]:
    return (
        TreeProblem(
            instance_id=instance_id,
            block_key=task[1],
            terminals=task[2],
            adj_map=component_data[task[0]]["adj_map"],
            node_weight_map=node_weight_map,
            node_index_map=component_data[task[0]]["node_index_map"],
            dimacs_id_map=component_data[task[0]]["dimacs_id_map"],
            inv_dimacs_id_map=component_data[task[0]]["inv_dimacs_id_map"],
            enable_super_root_index=task[3],
            do_debug=do_debug,
            mip_validation=mip_validation,
        )
        for task in tasks
    )


def _solve_treeproblem_tasks(
    problem_generator: Generator[TreeProblem],
    num_problems: int,
    results_dest: BlockResults,
    costs_dest: BlockCosts,
):
    """Solves the given tree problems switching between sequential and concurrent based on num_problems
    using the auto-switching `solve_tree` function which switches between DW and scipstp.
    """
    orig_num_results = len(results_dest)

    if num_problems >= TREE_PROBLEM_TASK_SEQUENTIAL_THRESHOLD:
        logger.warning("    _solve_treeproblem_tasks: solving concurrently...")

        try:
            with ProcessPoolExecutor(max_workers=TREE_PROBLEM_MAX_WORKERS) as executor:
                chunksize = max(
                    1, min(TREE_PROBLEM_MAX_CHUNKSIZE, num_problems // TREE_PROBLEM_TARGET_CHUNKS)
                )
                logger.warning(f"      chunksize: {chunksize}")

                # NOTE: solve_tree auto switches between DW and scipstp based on tree complexity
                results = executor.map(solve_tree, problem_generator, chunksize=chunksize)

                for completed_count, (block_key, cost, mask) in enumerate(results, start=1):
                    if completed_count % TREE_PROBLEM_PROGRESS_INTERVAL == 0:
                        logger.warning(
                            f"      solved ({completed_count}/{num_problems}) {completed_count / num_problems:.2%}..."
                        )

                    results_dest[block_key] = mask
                    costs_dest[block_key] = cost

        except BrokenProcessPool as e:
            logger.critical(f"Instance failed: Worker process died violently (OOM or Segfault). {e}")
            print(e)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Instance failed: Python exception bubbled up from worker: {e}")
            print(e, file=sys.stderr)

    else:
        logger.warning("    _solve_treeproblem_tasks: solving sequentially...")

        for block_n, problem in enumerate(problem_generator, start=1):
            logger.trace(f"    solving ({block_n}/{num_problems}) block {problem.block_key}...")

            # NOTE: solve_tree auto switches between DW and scipstp based on tree complexity
            block_key, cost, mask = solve_tree(problem)

            logger.trace(
                f"      solved ({block_n}/{num_problems}) block {block_key} containing {len(problem.terminals)} terminals with cost {cost}"
            )

            results_dest[block_key] = mask
            costs_dest[block_key] = cost

    logger.warning(f"      solved {len(results_dest) - orig_num_results} unique valid blocks ..")


def _generate_valid_partition_blocks(
    component_data: dict[int, ConnectedComponentMappings],
    interactivity_graph: PyDiGraph,
    blocked_edges: BlockedInteractionEdges,
) -> tuple[dict[ConnectedComponentMappingKey, set[BlockKey]], int, int]:
    """Generate structurally valid blocks from all connected components.
    Returns a mapping from connected component index to a set of valid blocks
    """
    logger.trace("    generating valid partition blocks...")

    valid_blocks: dict[ConnectedComponentMappingKey, set[BlockKey]] = defaultdict(set)
    num_blocks = 0
    num_candidate_blocks = 0

    blocked_block_keys = {tuple(sorted(edge)) for edge in blocked_edges}
    for cc_i, cc_data in component_data.items():
        reachable = sorted(cc_data.get("reachable"))
        k = len(reachable)
        for mask in range(1, 1 << k):
            block = tuple(reachable[i] for i in range(k) if mask & (1 << i))
            num_candidate_blocks += 1

            if len(block) == 2 and block in blocked_block_keys:
                continue

            subIG = interactivity_graph.subgraph(block)
            if not rx.is_strongly_connected(subIG):
                continue

            valid_blocks[cc_i].add(block)
            num_blocks += 1

    logger.warning(
        f"    generated {num_blocks} structurally valid blocks of {num_candidate_blocks} candidates..."
    )

    return valid_blocks, num_blocks, num_candidate_blocks


def _map_block_terminals(
    coverage_sets: CoverageSets,
    valid_blocks: dict[ConnectedComponentMappingKey, set[BlockKey]],
) -> dict[tuple[ConnectedComponentMappingKey, BlockKey], list[int]]:
    # Populate and map block terminals.
    block_terminals = {}
    for cc_i, blocks in valid_blocks.items():
        for block_key in blocks:
            terminals = set()
            for r in block_key:
                terminals.add(r)
                terminals.update(coverage_sets[r])
            block_terminals[(cc_i, block_key)] = sorted(terminals)

    return block_terminals


def _retain_dominant_blocks_by_singletons(
    block_results: BlockResults,
    block_costs: BlockCosts,
) -> tuple[BlockResults, BlockCosts]:
    logger.warning(f"    retaining dominant blocks by singletons... ({len(block_results)})")

    dominant_blocks = {}
    dominant_costs = {}

    for block_key, block_cost in block_costs.items():
        if sum(block_costs[(comp,)] for comp in block_key) < block_cost:
            continue

        dominant_blocks[block_key] = block_results[block_key]
        dominant_costs[block_key] = block_cost

    return dominant_blocks, dominant_costs


def _is_structurally_admissible(
    block: tuple[NodeIndex, ...],
    root_components: dict[NodeIndex, set[int]],
) -> bool:
    if len(block) <= 1:
        return True

    parent: dict[object, object] = {}

    def find(x: object) -> object:
        # ensure the key exists
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: object, y: object) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # link every root to the Steiner components it touches
    for r in block:
        find(r)  # make sure the root itself is present
        for c in root_components.get(r, ()):
            union(r, ("c", c))

    # all roots must share the same representative
    roots = list(block)
    rep = find(roots[0])
    return all(find(r) == rep for r in roots[1:])


def _retain_structurally_admissible_tasks(
    tasks: list[BlockTask],
    root_components: dict[NodeIndex, set[int]],
) -> list[BlockTask]:
    """Drop composite tasks whose roots cannot interact without leaving the block."""
    surviving: list[BlockTask] = []
    num_pruned = 0

    for task in tasks:
        _cc_i, block_key, _terminals, _sr_index = task

        if len(block_key) <= 1:
            surviving.append(task)
            continue

        if _is_structurally_admissible(block_key, root_components):
            surviving.append(task)
        else:
            num_pruned += 1

    logger.warning(
        f"    structural admissibility pruned {num_pruned} of {len(tasks)} composite tasks "
        f"({num_pruned / max(1, len(tasks)):.2%})"
    )
    return surviving


def _retain_structurally_admissible_blocks(
    valid_blocks: dict[ConnectedComponentMappingKey, set[BlockKey]],
    root_components: dict[NodeIndex, set[int]],
) -> dict[ConnectedComponentMappingKey, set[BlockKey]]:
    """Drop composite blocks whose roots cannot interact without leaving the block.

    Uses the pure-Steiner component labelling: a block is admissible only when
    the roots it contains, linked solely through the Steiner components they
    touch, form a single connected piece.
    """
    surviving: dict[ConnectedComponentMappingKey, set[BlockKey]] = defaultdict(set)
    num_pruned = 0
    num_composite = 0

    for cc_i, blocks in valid_blocks.items():
        for block_key in blocks:
            if len(block_key) <= 1:
                surviving[cc_i].add(block_key)
                continue

            num_composite += 1
            if _is_structurally_admissible(block_key, root_components):
                surviving[cc_i].add(block_key)
            else:
                num_pruned += 1

    logger.warning(
        f"    structural admissibility pruned {num_pruned} of {num_composite} composite blocks "
        f"({num_pruned / max(1, num_composite):.2%})"
    )
    return surviving


def _retain_dominant_composite_tasks_by_distance(
    tasks: list[BlockTask],
    block_costs: BlockCosts,
    terminal_to_terminal_distances: dict[tuple[NodeIndex, NodeIndex], Cost | float],
    super_root_index: SuperRootIndex,
) -> list[BlockTask]:
    """Flat-task analog of _retain_dominant_blocks_by_distance -- same OPT >= W_MST / (2 - 2/k)
    bound, operating directly on (block_key, terminals, sr_index) tuples instead of the
    cc_i-keyed valid_blocks/block_terminals shape. sr is stripped from terminals before the
    bound test since it's a zero-cost synthetic sink, not a real interior/terminal point.
    """
    # NOTE: This is only used in the _super pathway due to _super block handling.
    surviving = []
    num_dist_pruned = num_mst_pruned = 0

    for cc_i, block_key, terminals, sr_index in tasks:
        real_terminals = [t for t in terminals if t != super_root_index]
        singleton_sum = sum(block_costs[(comp,)] for comp in block_key)

        max_dist = max(
            terminal_to_terminal_distances[(ti, tj)]
            for i, ti in enumerate(real_terminals)
            for tj in real_terminals[i + 1 :]
        )
        if max_dist > singleton_sum:
            num_dist_pruned += 1
            continue

        mst_weight = metric_closure_mst_weight(real_terminals, terminal_to_terminal_distances)
        lower_bound = steiner_lower_bound(mst_weight, len(real_terminals))
        if lower_bound > singleton_sum:
            num_mst_pruned += 1
            continue

        surviving.append((
            cc_i,
            block_key,
            terminals,
            sr_index,
        ))

    num_pruned = num_dist_pruned + num_mst_pruned
    logger.warning(
        f"    pre-solve bound pruned {num_pruned} of {len(tasks)} composite tasks "
        f"({num_dist_pruned} via max-dist, {num_mst_pruned} via MST, {num_pruned / max(1, len(tasks)):.2%})"
    )
    return surviving


def _retain_dominant_blocks_by_distance(
    valid_blocks: dict[ConnectedComponentMappingKey, set[BlockKey]],
    block_terminals: dict[tuple[ConnectedComponentMappingKey, BlockKey], TerminalsList],
    primitive_block_costs: BlockCosts,
    terminal_to_terminal_distances: dict[tuple[NodeIndex, NodeIndex], Cost | float],
) -> list[BlockTask]:
    logger.warning(f"    retaining dominant blocks by distance... ({len(valid_blocks)})")
    # NOTE: This should only be used on non-super solve as tree pathways as it sets the task
    #       super_root_index to zero (which is disabled)!

    # NOTE: OPT >= W_MST / (2 - 2/k) (KMB). If that lower bound already exceeds the cheapest
    #       decomposition we can prove right now (singleton sum), no joint solve can beat it.
    #       Strict '>' only -- ties fall through to the solver.
    #       (Same as the post-solve dominance check, since a tying composite may still share
    #       more structure with the rest of the forest than any decomposition would.)
    surviving_tasks = []
    num_dist_pruned = 0
    num_mst_pruned = 0

    for cc_i, blocks in valid_blocks.items():
        for block_key in blocks:
            if len(block_key) == 1:
                continue

            terminals = block_terminals[(cc_i, block_key)]
            singleton_sum = sum(primitive_block_costs[(r,)] for r in block_key)

            max_dist = max(
                terminal_to_terminal_distances[(ti, tj)]
                for i, ti in enumerate(terminals)
                for tj in terminals[i + 1 :]
            )

            if max_dist > singleton_sum:
                num_dist_pruned += 1
                continue

            mst_weight = metric_closure_mst_weight(terminals, terminal_to_terminal_distances)
            lower_bound = steiner_lower_bound(mst_weight, len(terminals))

            if lower_bound > singleton_sum:
                num_mst_pruned += 1
                continue

            surviving_tasks.append((cc_i, block_key, terminals, 0))

    num_bound_pruned = num_dist_pruned + num_mst_pruned
    num_composite_blocks = sum(1 for blocks in valid_blocks.values() for bk in blocks if len(bk) > 1)
    logger.warning(
        f"    pre-solve bound pruned {num_bound_pruned} of {num_composite_blocks} composite blocks "
        f"({num_dist_pruned} via max-dist, {num_mst_pruned} via MST, "
        f"{num_bound_pruned / max(1, num_composite_blocks):.2%})"
    )

    return surviving_tasks


def _blocks_to_blockmasks(
    coverage_representatives: CoverageRepresentatives,
    block_results: BlockResults,
    block_costs: BlockCosts,
) -> tuple[MaskedBlockSolutionMasks, MaskedBlockCosts]:
    logger.warning(f"    building block masks... ({len(block_results)})")

    coverage_bit = {r: 1 << i for i, r in enumerate(coverage_representatives)}

    block_mask_costs = {}
    block_mask_solutions = {}

    for block_key, solution_mask in block_results.items():
        block_mask = 0

        for r in block_key:
            block_mask |= coverage_bit[r]

        block_mask_costs[block_mask] = block_costs[block_key]
        block_mask_solutions[block_mask] = solution_mask

    return block_mask_solutions, block_mask_costs


def _blockmasks_to_blocks(
    coverage_representatives: CoverageRepresentatives,
    block_mask_solutions: MaskedBlockSolutionMasks,
    block_mask_costs: MaskedBlockCosts,
) -> tuple[BlockResults, BlockCosts]:
    logger.warning(f"    reversing block masks... ({len(block_mask_solutions)})")

    coverage_roots = sorted(coverage_representatives)

    block_results = {}
    block_costs = {}

    for block_mask, solution_mask in block_mask_solutions.items():
        block_key_list = []

        for i, r in enumerate(coverage_roots):
            if block_mask & (1 << i):
                block_key_list.append(r)

        block_key = tuple(block_key_list)

        block_results[block_key] = solution_mask
        block_costs[block_key] = block_mask_costs[block_mask]

    return block_results, block_costs


def _retain_dominant_masks_by_composite(
    block_mask_solutions: MaskedBlockSolutionMasks,
    block_mask_costs: MaskedBlockCosts,
) -> tuple[MaskedBlockSolutionMasks, MaskedBlockCosts]:
    logger.warning(f"    retaining dominant masks by composite DP... ({len(block_mask_costs)})")

    dominant_costs = {}
    dominant_solutions = {}

    for block_mask in sorted(block_mask_costs, key=int.bit_count):
        direct_cost = block_mask_costs[block_mask]

        # Singletons are always primitive.
        if block_mask.bit_count() == 1:
            dominant_costs[block_mask] = direct_cost
            dominant_solutions[block_mask] = block_mask_solutions[block_mask]
            continue

        best_decomposition = float("inf")

        sub = (block_mask - 1) & block_mask
        while sub:
            other = block_mask ^ sub

            if sub < other and sub in dominant_costs and other in dominant_costs:
                decomposition_cost = dominant_costs[sub] + dominant_costs[other]
                best_decomposition = min(best_decomposition, decomposition_cost)

            sub = (sub - 1) & block_mask

        # Keep only primitive realizations.
        if best_decomposition >= direct_cost:
            dominant_costs[block_mask] = direct_cost
            dominant_solutions[block_mask] = block_mask_solutions[block_mask]

    return dominant_solutions, dominant_costs


def _solve_composite_blocks_dp(
    G: PyDiGraph,
    coverage_representatives: CoverageRepresentatives,
    block_results: BlockResults,
    block_costs: BlockCosts,
) -> CompositeSolution:
    logger.warning(f"    _solve_composite_blocks_dp... ({len(block_results)})")
    block_results, block_costs = _retain_dominant_blocks_by_singletons(block_results, block_costs)

    block_mask_solutions, block_mask_costs = _blocks_to_blockmasks(
        coverage_representatives, block_results, block_costs
    )
    block_mask_solutions, block_mask_costs = _retain_dominant_masks_by_composite(
        block_mask_solutions, block_mask_costs
    )

    coverage_roots = sorted(coverage_representatives)
    coverage_bit = {r: 1 << i for i, r in enumerate(coverage_roots)}

    best_solution_mask, best_cost = _solve_partitioned_tree_dp(
        set(coverage_representatives), coverage_bit, block_mask_costs, block_mask_solutions
    )
    best_solution = _unmask_solution(G, best_solution_mask)

    return best_solution, int(best_cost), block_results, block_costs


def _partition_block_masks_by_cooccurrence(
    coverage_representatives: CoverageRepresentatives,
    coverage_bits: CoverageBits,
    block_mask_costs: MaskedBlockCosts,
) -> list[tuple[dict[int, int], dict[int, int], list[int]]]:
    """Groups coverage representatives into independent components based on which
    representatives actually co-occur in some surviving block mask (post-dominance
    filtering), then regroups block_mask_costs per component with compact local bits.

    Two representatives only ever need a joint DP state if some surviving block
    contains both, directly or transitively via a chain of shared blocks. Everything
    outside that union-find grouping is provably independent -- the global optimum
    decomposes exactly into each component's independent optimum, so a single
    combined n-bit DP pays for state combinations no actual block could realize.
    """
    bit_to_rep = {b: r for r, b in coverage_bits.items()}
    parent = {r: r for r in coverage_representatives}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for mask in block_mask_costs:
        bits, m = [], mask
        while m:
            low = m & -m
            bits.append(bit_to_rep[low])
            m ^= low
        for b in bits[1:]:
            union(bits[0], b)

    groups: dict[int, list[int]] = defaultdict(list)
    for r in coverage_representatives:
        groups[find(r)].append(r)
    component_groups = [sorted(g) for g in groups.values()]

    global_to_local: dict[int, tuple[int, int]] = {}
    for c_i, group in enumerate(component_groups):
        for i, r in enumerate(group):
            global_to_local[coverage_bits[r]] = (c_i, 1 << i)

    per_component_costs: list[dict[int, int]] = [{} for _ in component_groups]
    per_component_l2g: list[dict[int, int]] = [{} for _ in component_groups]

    for global_mask, cost in block_mask_costs.items():
        local_mask, owner, m = 0, None, global_mask
        while m:
            low = m & -m
            c_i, lb = global_to_local[low]
            owner = c_i
            local_mask |= lb
            m ^= low

        assert owner is not None
        per_component_costs[owner][local_mask] = cost
        per_component_l2g[owner][local_mask] = global_mask

    return list(zip(per_component_costs, per_component_l2g, component_groups))


def _solve_partitioned_tree_dp(
    coverage_representatives: CoverageRepresentatives,
    coverage_bits: CoverageBits,
    block_mask_costs: MaskedBlockCosts,
    block_mask_solutions: MaskedBlockSolutionMasks,
) -> tuple[SolutionMask, Cost | float]:
    partitioned = _partition_block_masks_by_cooccurrence(
        coverage_representatives, coverage_bits, block_mask_costs
    )

    if len(partitioned) > 1:
        logger.warning(f"    partition DP split into {len(partitioned)} independent component(s)")

    best_solution_mask = 0
    total_cost: int | float = 0

    start_time = time()

    for local_costs, local_to_global, group in partitioned:
        n_c = len(group)
        if n_c == 0 or not local_costs:
            continue

        local_solutions = {local: block_mask_solutions[g] for local, g in local_to_global.items()}

        dp, choice = _solve_tree_partitions_dp(n_c, local_costs)
        comp_solution_mask, comp_cost = _extract_dp_solution(n_c, local_solutions, dp, choice)

        if comp_cost == float("inf"):
            logger.error(f"    no valid partition solution for component of size {n_c}")
            return 0, float("inf")

        total_cost += comp_cost
        best_solution_mask |= comp_solution_mask

    logger.warning(f"      partition DP took {time() - start_time:.3f} seconds to solve")

    return best_solution_mask, total_cost


def _solve_tree_partitions_dp(
    num_coverage_representatives: int,
    block_mask_costs: MaskedBlockCosts,
) -> tuple[list[int | float], list[tuple[int, int] | None]]:
    logger.trace(f"      solving tree partitions DP... ({len(block_mask_costs)})")

    start_time = time()

    n = num_coverage_representatives
    full_mask = (1 << n) - 1

    dp = [float("inf")] * (full_mask + 1)
    choice = [None] * (full_mask + 1)

    dp[0] = 0

    # Two traversal strategies for the same subset-sum DP, with very different cost
    # profiles depending on how many blocks survived Phase 2's dominance filter:
    #
    #   - submask enumeration: Theta(3^n) total candidate_submasks iterations,
    #                          independent of |block_mask_costs|.
    #     Cheap when block_mask_costs is dense (most submasks are real hits).
    #   - block iteration:     Theta(2^n * |block_mask_costs|) total iterations.
    #     Cheap when block_mask_costs is sparse relative to the full state space.
    #
    # Crossover: 2^n * B == 3^n  =>  B == (3/2)^n. Pick whichever side is smaller
    # instead of assuming one regime; the dominance filter's yield varies a lot
    # across instances and can't be assumed sparse from a handful of samples.
    num_blocks = len(block_mask_costs)
    crossover = 1.5**n

    use_block_iteration = num_blocks < crossover

    logger.debug(
        f"        n={n}, |block_mask_costs|={num_blocks}, crossover={crossover:.1f} "
        f"-> using {'block-iteration' if use_block_iteration else 'submask-enumeration'} pathway"
    )

    state_transitions = 0
    candidate_checks = 0

    if use_block_iteration:
        block_items = list(block_mask_costs.items())

        for state in range(full_mask + 1):
            if dp[state] == float("inf"):
                continue

            for block_mask, block_cost in block_items:
                candidate_checks += 1
                if block_mask & state:
                    continue

                next_state = state | block_mask
                new_cost = dp[state] + block_cost
                state_transitions += 1

                if new_cost < dp[next_state]:
                    dp[next_state] = new_cost
                    choice[next_state] = (state, block_mask)

    else:
        for state in range(full_mask + 1):
            if dp[state] == float("inf"):
                continue

            remaining = full_mask ^ state
            sub = remaining

            while sub:
                candidate_checks += 1
                if sub in block_mask_costs:
                    new_cost = dp[state] + block_mask_costs[sub]

                    next_state = state | sub
                    state_transitions += 1

                    if new_cost < dp[next_state]:
                        dp[next_state] = new_cost
                        choice[next_state] = (state, sub)

                sub = (sub - 1) & remaining

    logger.debug(
        f"        solved {candidate_checks} candidate checks of {state_transitions} state transitions in {time() - start_time:.2f}s"
    )

    return dp, choice


def _extract_dp_solution(
    num_coverage_representative_sets: int,
    block_mask_solutions: dict[BlockMask, SolutionMask],
    dp: list[int | float],
    choice: list[tuple[int, int] | None],
) -> tuple[SolutionMask, Cost | float]:
    logger.trace("    extracting best solution from DP...")

    full_mask = (1 << num_coverage_representative_sets) - 1
    best_cost = dp[full_mask]
    best_solution_mask = 0

    if best_cost == float("inf") or choice[full_mask] is None:
        logger.error("    no valid partition solution found")
        return 0, float("inf")

    state = full_mask

    while state:
        prev_choice = choice[state]
        if prev_choice is None:
            logger.error(f"    broken DP reconstruction at state {state}")
            break

        prev_state, block_mask = prev_choice
        best_solution_mask |= block_mask_solutions[block_mask]
        state = prev_state

    return best_solution_mask, best_cost


def _unmask_solution(
    G: PyDiGraph,
    mask: SolutionMask,
    nodes_list: list[NodeIndex] | None = None,
) -> set[int]:
    solution = set()

    if nodes_list is None:
        nodes_list = list(G.node_indices())

    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        u = nodes_list[idx]
        solution.add(u)
        mask ^= lsb

    return solution


def steiner_lower_bound(mst_weight, num_terminals):
    """
    OPT >= w(MST) / (2 - 2/l), where l = # leaves of OPT.
    Since every leaf of an optimal Steiner tree is a terminal, l <= num_terminals,
    and (2 - 2/l) is increasing in l, so substituting num_terminals for the
    unknown true l only loosens (never invalidates) the bound.
    """
    if num_terminals <= 1:
        return 0.0
    ratio = 2 - (2 / num_terminals)
    return mst_weight / ratio


def metric_closure_mst_weight(
    terminals: list[int], terminal_to_terminal_distances: dict[tuple[int, int], float]
):
    """Kruskal over the complete graph implied by pairwise terminal distances."""
    terminals = sorted(terminals)
    if len(terminals) <= 1:
        return 0.0

    edges = sorted(
        (
            (terminal_to_terminal_distances[(ti, tj)], ti, tj)
            for i, ti in enumerate(terminals)
            for tj in terminals[i + 1 :]
        ),
        key=lambda e: e[0],
    )

    parent = {t: t for t in terminals}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    weight, joined = 0.0, 0
    for w, ti, tj in edges:
        ri, rj = find(ti), find(tj)
        if ri != rj:
            parent[ri] = rj
            weight += w
            joined += 1
            if joined == len(terminals) - 1:
                break
    return weight
