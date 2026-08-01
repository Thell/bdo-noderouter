from __future__ import annotations

import heapq
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from itertools import combinations
from time import time

import networkx as nx
import nwst_dw
import rustworkx as rx
from bidict import bidict
from loguru import logger
from rustworkx import PyDiGraph
from sfpgre_cch import get_interactivity_edges

from api_common import PAYLOAD_WEIGHT_KEY
from api_exploration_data import get_exploration_data
from api_nwstp_problem import ConnectedComponentMappings, TreeProblem
from api_nwstp_solver import solve_tree
from api_rx_pydigraph import subgraph_stable

HYPERNODE_CONTENTS_KEY = "collapsed_nodes"
MAX_ENCLOSED_STEINER_INTERFACES = 6

DW_SOLVE_PAR_THRESHOLD = 16  # This will never trigger when the scipstp setup is done
DW_MAX_TREE_TERMINALS = 6
DW_MAX_TREE_COMPLEXITY = 8_000_000_000

# NOTE: The time sink in solving as trees is the partition block filtering.
# When there is an unreachable root the partitions need to be enumerated and
# this follows the Bell number pattern per root.
SCIPSTP_PAR_ROOTS_THRESHOLD = 8  # This will have 2^{n}-1 partition blocks to solve
SCIPSTP_MAX_PAR_ROOTS = 13  # This will have 2^{n}-1 partition blocks to solve
SCIPSTP_MAX_ROOTS = 6  # Beyond this the problem will be sent to MIP
SCIPSTP_NPERR_THRESHOLD = 15  # Beyond this the problem will be sent to MIP


r"""
# Steiner Forest Reduction Grammar (v1.0)

The purpose of this document is to facilitate the communication of various
reductions that can be utilized during pre-processing of a Steiner forest
problem prior to approximation or MIP solving that retains the optimal solution.

In the absence of any super root the utilized graph is a node weighted,
undirected graph with edges represented as bi-directional arcs with no weight.
When a super root is present it is expected that the resulting problem will be
modelled or solved in such a way that a super root is a sink and super terminals
are normal nodes within the graph acting as sources that can satisfy super root
demand via any potential root. As such, a super root only has inbound edges and
potential roots have an outbound edge to the super root. All other connections
are symmetric.

## 1. Symbol Glossary

### Nodes & Sets

| Symbol | Meaning | Notes |
| :--- | :--- | :--- |
| **○** | Neutral Placeholder | Used for general neighbors in abstract rules. |
| **Ⓢ** | Steiner Node | An optional node; no inherent weight/demand. |
| **Ⓣ** | Terminal Node | A node that *must* be connected to its partner. |
| **Ⓡ** | Root Node | The root of an Arborescence (**𝓐**). |
| **𝕡** | Potential Root Node | An individual node member of **𝔹**. |
| **𝕥** | Super Terminal Node | A terminal whose demand is met by any node in **𝔹**. |
| **ⓤ, ⓥ, ⓦ** | Variable Nodes | Generic nodes often consumed in reductions. |
| **⦿** | Hyper-Node | A contracted component or cluster of nodes. |
| **𝓐𐞪** | Arborescence Set | The cluster of nodes rooted at **Ⓡ**. |
| **𝓟** | Potential Roots Set | Set of nodes with edges to a Super Root. |
| **𝓡** | Roots Set | Set of nodes rooting the forest of trees in 𝓐. |
| **𝕊** | Super Root Set | A "sink" for potential root demands. |
| **𝐆** | Graph Set | Active nodes and edges. |
| **𝐃** | Demand Set | Unsatisfied terminal pairs. |
| **𝑻** | Terminal Sets | Unsatisfied `root: {terminal}` sets. |
| **𝓢** | Solution Set | Nodes/Edges selected for the forest. |

### Connectivity & Topology

| Symbol | Meaning | Notes |
| :--- | :--- | :--- |
| **⇋** | Bi-directional Edge | Parallel edges or symmetric relationships. |
| **🡒** | Specific Path | A single, defined path between two nodes. |
| **↠** | Existence of Path | Represents any or all paths between nodes. |
| **↠ⓥ↠** | Existence of Path Traversing Variable Node | Represents any or all such paths between nodes. |
| **↠∣ⓥ∣↠** | Existence of Path Avoiding Variable Node | Represents any or all such paths between nodes. |
| **∣** | Boundary | Absolute boundary. |
| **⁝** | Universe Boundary | The "Tricolon" isolation cut. |
| **⋯** | Continuity | Indicates a path extends further into the graph. |
| **∞** | Infinite Set | The global "Universe" beyond a node. |

### Logic & Operators

| Symbol | Meaning | Role |
| :--- | :--- | :--- |
| **⭆** | Transform | The multi-operation reduction arrow. |
| **:** | Guard Delimiter | "Where / Such that" for local node properties. |
| **⇔** | Invariant | "If and only if" for global/attribute constraints. |
| **∈ / ∉** | Membership | Relative to an Arborescence **𝓐** or Set. |
| **∖ / ⋃** | Set Ops | Exclusion (subtraction) and Union (addition). |
| **𝔀( )** | Weight Function | The cost of a node or path. |
| **( )** | Degree Function | The degree of a node. |

---

## 2. Grammar Structure
Rules follow a "Pattern-First" sequence:

**`|Pattern : Local Guards ⭆ Transform ⇔ Global/Attribute Guards`**

---

## Definitions

### Steiner node

Any node which exists in the graph and is not a part of the solution.
In the presence of a super root node specified Steiner nodes may be classified
as a potential root for super terminals.

> Ⓢ ≔ ⓥ ∈ {𝐆 ∖ { 𝓢 ⋃ 𝓟 }}

### Terminal node

Any node which exists in the graph and is a part of the solution.
A specially designated terminal is classified as a root for each terminal
set cluster which will span the arborsence rooted at that specified node.

> Ⓣ ≔ ⓥ ∈ { 𝓢 ∖ 𝓡 }

### Root node

Any node designated as the root of an arborescence (rooted tree) of the
active demand sets as denoted by 𝓐𐞪 excluding any super root.

> Ⓡ ≔ ⓥ ∈ { 𝓡 ∖ 𝕊 }
"""

# Primitive Types
type BlockKey = tuple[int, ...]
type BlockMask = int
type Cost = int
type ConnectedComponentMappingKey = int
type CoverageRepresentative = int
type SolutionMask = int

# Collection Types
type CoverageRepresentatives = set[int]
type CoverageSets = dict[int, set[int]]
type SolutionSet = set[int]
type TerminalsList = list[int]
type TerminalsSet = set[int]
type BlockCosts = dict[BlockKey, Cost]
type BlockResults = dict[BlockKey, SolutionMask]
type MaskedBlockCosts = dict[BlockMask, Cost]
type MaskedBlockResults = dict[BlockMask, BlockResults]
type MaskedBlockSolutionMasks = dict[BlockMask, SolutionMask]

# Tuple Types
type BlockTask = tuple[ConnectedComponentMappingKey, BlockKey, TerminalsList]
type CompositeSolution = tuple[SolutionSet, Cost, BlockResults, BlockCosts]


def _connected_component_mappings(
    G: PyDiGraph, coverage_representatives: set[int], sr_index: int | None = None
) -> tuple[dict[int, int], dict[int, ConnectedComponentMappings]]:
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
        if sr_index is not None and G.has_node(sr_index)
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

    logger.warning(
        f"      identified {len(component_data)} reachability component(s): {[len(cc_data.get('reachable')) for cc_data in component_data.values()]}"
    )

    return representatives_component_map, component_data


def _interactivity_graph(
    G: PyDiGraph,
    coverage_sets: CoverageSets,
) -> PyDiGraph:
    """Creates the interactivity graph for the given reduced state graph over the given coverage sets."""
    coverage_representatives = set(coverage_sets.keys())
    node_weight_map = {u: G[u][PAYLOAD_WEIGHT_KEY] for u in G.node_indices()}

    # Union sets of all nodes for all shortest paths for each t -> r for each t in terminal_set r
    arbor_rt_all_shortest_path_unions = {r: set() for r in coverage_representatives}
    for r, terminals in coverage_sets.items():
        if r not in coverage_representatives:
            continue
        for t in terminals:
            paths = rx.all_shortest_paths(G, t, r, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
            arbor_rt_all_shortest_path_unions[r].update(*paths)

    # TODO: check how the interactivity edges handle the presence of super root...
    interactivity_edges = get_interactivity_edges(
        G, coverage_sets, node_weight_map, arbor_rt_all_shortest_path_unions
    )

    interactivity_graph = subgraph_stable(G, coverage_representatives)
    interactivity_graph.remove_edges_from(interactivity_graph.edge_list())
    interactivity_graph.add_edges_from(interactivity_edges)

    return interactivity_graph


def _problem_generator(
    instance_id: str,
    component_data: dict[ConnectedComponentMappingKey, ConnectedComponentMappings],
    node_weight_map: dict[int, int],
    tasks: list[BlockTask],
    sr_index: int = 0,
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
            enable_super_root_index=sr_index,
            do_debug=do_debug,
            mip_validation=mip_validation,
        )
        for task in tasks
    )


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

    num_coverage_representatives = len(coverage_representatives)

    dp, choice = _solve_tree_partitions_dp(num_coverage_representatives, block_mask_costs)

    best_solution_mask, best_cost = _extract_dp_solution(
        num_coverage_representatives, block_mask_solutions, dp, choice
    )
    best_solution = _unmask_solution(G, best_solution_mask)

    return best_solution, int(best_cost), block_results, block_costs


def _generate_valid_partition_blocks(
    component_data: dict[int, ConnectedComponentMappings], interactivity_graph: PyDiGraph
) -> tuple[dict[int, set[BlockKey]], int, int]:
    """Generate structurally valid blocks from all connected components.
    Returns a mapping from connected component index to a set of valid blocks
    """
    logger.trace("    generating valid partition blocks...")

    valid_blocks: dict[int, set[BlockKey]] = defaultdict(set)
    num_blocks = 0
    num_candidate_blocks = 0

    for cc_i, cc_data in component_data.items():
        reachable = sorted(cc_data.get("reachable"))
        k = len(reachable)
        for mask in range(1, 1 << k):
            block = tuple(reachable[i] for i in range(k) if mask & (1 << i))
            num_candidate_blocks += 1

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
    coverage_sets: dict[int, set[int]], valid_blocks: dict[int, set[tuple]]
) -> dict[tuple[int, tuple[int, ...]], list[int]]:
    # Populate and map block terminals.
    block_terminals = {}
    for cc_i, blocks in valid_blocks.items():
        for block_key in blocks:
            terminals = set()
            for r in block_key:
                terminals.add(r)

                # TODO: instead of using terminal_sets directly here, we should
                #       use the representative's cover set, since that might be
                #       different when handling super terminal cases...
                #       Until the refactor is complete we'll use terminal_sets...
                terminals.update(coverage_sets[r])
            block_terminals[(cc_i, block_key)] = sorted(terminals)

    return block_terminals


def _retain_dominant_blocks_by_singletons(
    block_results: BlockResults, block_costs: BlockCosts
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


def _retain_dominant_composite_tasks_by_distance(
    tasks: list[tuple[tuple[int, ...], list[int], int]],  # (block_key, terminals, sr_index)
    block_costs: dict[tuple[int, ...], int],
    terminal_to_terminal_distances: dict[tuple[int, int], int | float],
    sr: int,
) -> list[tuple[tuple[int, ...], list[int], int]]:
    """Flat-task analog of _retain_dominant_blocks_by_distance -- same OPT >= W_MST / (2 - 2/k)
    bound, operating directly on (block_key, terminals, sr_index) tuples instead of the
    cc_i-keyed valid_blocks/block_terminals shape. sr is stripped from terminals before the
    bound test since it's a zero-cost synthetic sink, not a real interior/terminal point.
    """
    # NOTE: This is currently only used in the _super pathway until unification is complete
    #      because of how the block handling differs from normal.
    surviving = []
    num_dist_pruned = num_mst_pruned = 0

    for block_key, terminals, sr_index in tasks:
        real_terminals = [t for t in terminals if t != sr]
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

        surviving.append((block_key, terminals, sr_index))

    num_pruned = num_dist_pruned + num_mst_pruned
    logger.warning(
        f"    pre-solve bound pruned {num_pruned} of {len(tasks)} composite tasks "
        f"({num_dist_pruned} via max-dist, {num_mst_pruned} via MST, {num_pruned / max(1, len(tasks)):.2%})"
    )
    return surviving


def _retain_dominant_blocks_by_distance(
    valid_blocks: dict[int, set[BlockKey]],
    block_terminals: dict[tuple[int, tuple[int, ...]], list[int]],
    primitive_block_costs: dict[tuple[int], int],
    terminal_to_terminal_distances: dict[tuple[int, int], int | float],
) -> list[BlockTask]:
    logger.warning(f"    retaining dominant blocks by distance... ({len(valid_blocks)})")

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

            surviving_tasks.append((cc_i, block_key, terminals))

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
    coverage_sets: dict[int, set[int]],
    block_mask_solutions: dict[int, int],
    block_mask_costs: dict[int, int],
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]:
    logger.warning(f"    reversing block masks... ({len(block_mask_solutions)})")

    coverage_roots = sorted(coverage_sets.keys())

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
    block_mask_solutions: MaskedBlockSolutionMasks, block_mask_costs: MaskedBlockCosts
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


def _solve_tree_partitions_dp(
    num_coverage_representatives: int, block_mask_costs: MaskedBlockCosts
) -> tuple[list[int | float], list[tuple[int, int] | None]]:
    logger.warning(f"    solving tree partitions DP... ({len(block_mask_costs)})")

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

    logger.warning(
        f"    n={n}, |block_mask_costs|={num_blocks}, crossover={crossover:.1f} "
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

    logger.warning(
        f"    solved {candidate_checks} candidate checks of {state_transitions} state transitions in {time() - start_time:.2f}s"
    )

    return dp, choice


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
    coverage_representatives: set[int],
    coverage_bit: dict[int, int],
    block_mask_costs: dict[int, int],
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
    bit_to_rep = {b: r for r, b in coverage_bit.items()}
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
            global_to_local[coverage_bit[r]] = (c_i, 1 << i)

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
    coverage_representatives: set[int],
    coverage_bit: dict[int, int],
    block_mask_costs: dict[int, int],
    block_mask_solutions: dict[int, int],
) -> tuple[int, int | float]:
    partitioned = _partition_block_masks_by_cooccurrence(
        coverage_representatives, coverage_bit, block_mask_costs
    )

    if len(partitioned) > 1:
        logger.warning(f"    partition DP split into {len(partitioned)} independent component(s)")

    total_cost: int | float = 0
    best_solution_mask = 0

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

    return best_solution_mask, total_cost


def _extract_dp_solution(
    num_coverage_representativesge_sets: int,
    block_mask_solutions: dict[int, int],
    dp: list[int | float],
    choice: list[tuple[int, int] | None],
) -> tuple[int, int | float]:
    logger.trace("    extracting best solution from DP...")

    full_mask = (1 << num_coverage_representativesge_sets) - 1
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


def _unmask_solution(G: PyDiGraph, mask: int, nodes_list: list[int] | None = None) -> set[int]:
    logger.trace("    extracting solution from mask...")
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


@dataclass
class SFGraphReductionEngine:
    """
    A reduction engine for the Steiner Forest problem.

    Reduces the problem space of the Steiner Forest problem solution for a multi-commodity flow problem
    focusing on reducing the problem graph which reduces MIP model variables, constraints and non-zeros.

    ** All reductions and transformations guarantee that the remaining problem space contains an optimal solution. **

    With that in mind, only certified inclusion/exclusion/redundancy is a consideration.
    """

    instance_id: str
    graph: PyDiGraph
    node_to_key: bidict[int, int]
    super_root_index: int | None = None

    fixed_nodes: set[int] = field(default_factory=set)
    potential_roots: set[int] = field(default_factory=set)
    terminal_sets: dict[int, set[int]] = field(default_factory=dict)

    _seen_nonreducible_outer_windows: set[tuple[int, ...]] = field(default_factory=set)
    _seen_nonreducible_steiner_face_clusters: set[tuple[int, ...]] = field(default_factory=set)

    # Global maximum distance between a root and its' furthest terminal
    _global_min_max_drt: float = float("inf")

    # solve_as_tree is a witness of a non-reducible graph and should only execute once
    _solved_as_tree: bool = False

    do_debug: bool = False
    call_counts: Counter = field(default_factory=Counter)
    solved_trees = 0

    # MARK: Common Helpers

    # Logging/Debugging helpers
    def dump_graph(self, msg: str, graph: PyDiGraph | None = None, override_debug: bool = False):
        if not self.do_debug and not override_debug:
            return
        print(f"=== BEGIN: {msg} ===")
        G = graph if graph is not None else self.graph
        print(f"    num_nodes = {G.num_nodes()}, num_edges = {G.num_edges()}\n")
        terminals = {
            self.get_node_key(t): self.get_node_key(r) for r, ts in self.terminal_sets.items() for t in ts
        }
        edges = [(self.get_node_key(s), self.get_node_key(t)) for s, t in G.edge_list()]
        weights = {self.get_node_key(n): G[n][PAYLOAD_WEIGHT_KEY] for n in G.node_indices()}

        print(f"    terminals = {terminals}")
        print(f"    actual = { {self.get_node_key(n) for n in G.node_indices()} }")
        print(f"    actual_edges = {edges}")
        print(f"    actual_weights = {weights}")
        print(f"=== END: {msg} ===")

    def dump_state(self, msg: str, trigger_count: int = 0):
        self.call_counts[msg] += trigger_count
        logger.debug(f"Post {msg}:  trigger count: {trigger_count}")
        if trigger_count == 0:
            return

        sets = self.terminal_sets
        logger.debug(
            f"  Nodes: {self.graph.num_nodes():4}, Edges: {self.graph.num_edges():4}, Roots: {len(sets):4}, Terminals: {sum(len(ts) for ts in sets.values()):4}"
        )
        sets_wp = {self.get_node_key(r): {self.get_node_key(t) for t in ts} for r, ts in sets.items()}
        logger.debug(f"  Demand sets: {sets_wp}")

    def dump_final_report(
        self,
        reduced_root_pairs_wp: dict[int, int],
        fixed_nodes_wp: set[int],
    ):
        print("=== Post final reduction ===")
        logger.debug(f"  {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges")
        logger.debug(f"  {len(reduced_root_pairs_wp)} pairs")
        logger.trace(f"  {reduced_root_pairs_wp=}")
        logger.trace(f"  {fixed_nodes_wp=}")
        logger.trace(f"  num components: {rx.number_strongly_connected_components(self.graph)}")

        node_weights = [n[PAYLOAD_WEIGHT_KEY] for n in self.graph.nodes()]
        nodes_by_weight = Counter(node_weights)
        nodes_by_weight = sorted(nodes_by_weight.items())
        logger.debug(f"  nodes by weight: {nodes_by_weight}")

        node_degrees = [self.graph.out_degree(n) for n in self.graph.node_indices()]
        nodes_by_degree = Counter(node_degrees)
        nodes_by_degree = sorted(nodes_by_degree.items())
        logger.debug(f"  nodes by degree: {nodes_by_degree}")

        # Breakdown by degree into fixed, basetown, and steiner
        towns = get_exploration_data().towns
        for d, c in nodes_by_degree:
            sub_counter = Counter()
            for i in self.graph.node_indices():
                is_town = self.get_node_key(i) in towns
                if self.graph.out_degree(i) == d:
                    if i in self.fixed_nodes:
                        sub_counter["fixed"] += 1
                    if is_town:
                        sub_counter["basetown"] += 1
                    if i not in self.fixed_nodes and not is_town:
                        sub_counter["steiner"] += 1
            logger.trace(f"    {d}: {sub_counter}")

        logger.trace("=== PATHS ===")
        reduced_root_pairs: dict[int, int] = {}
        for r, comp in self.terminal_sets.items():
            for t in comp:
                reduced_root_pairs[t] = r
        for u, v in reduced_root_pairs.items():
            has_u = self.graph.has_node(u)
            has_v = self.graph.has_node(v)
            has_path = rx.has_path(self.graph, u, v) if has_u and has_v else False
            logger.trace(
                f"  {self.get_node_key(u)} ({has_u}) -> {self.get_node_key(v)} ({has_v}) => {has_path}"
            )

        logger.debug("\n=== REDUCTIONS ===")
        assert self.call_counts
        logger.debug(f"Total reduction trigger count: {sum(self.call_counts.values())}")
        logger.debug("Trigger counts per reduction step:")
        for msg, count in sorted(self.call_counts.items()):
            logger.debug(f"  {msg}: {count}")

        logger.debug("=== STEINERS ===")
        nodes = set(self.graph.node_indices())
        non_steiner_nodes = set(self.non_steiner_nodes())
        steiner_nodes = nodes - non_steiner_nodes
        logger.debug(f"  {len(steiner_nodes)} steiner nodes")

        critical_steiner_nodes = self.critical_steiner_nodes()
        if critical_steiner_nodes:
            logger.warning(f"  {len(critical_steiner_nodes)} critical steiner nodes")
            logger.warning(f"  {critical_steiner_nodes=}")

    def dump_reduction_results(
        self,
        msg: str,
        num_edges_start: int,
        num_nodes_start: int,
        num_terminal_roots_start: int,
        num_terminals_start: int,
    ):
        # Reduction percentages
        num_edges_end = self.graph.num_edges()
        num_nodes_end = self.graph.num_nodes()
        num_terminal_roots_end = len(self.terminal_sets)
        num_terminals_end = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_end
        per_edges = (num_edges_end - num_edges_start) / num_edges_start
        per_nodes = (num_nodes_end - num_nodes_start) / num_nodes_start
        per_terminals = (num_terminals_end - num_terminals_start) / num_terminals_start
        per_roots = (num_terminal_roots_end - num_terminal_roots_start) / num_terminal_roots_start
        num_super_terminals = (
            0 if self.super_root_index is None else len(self.terminal_sets.get(self.super_root_index, []))
        )
        print(
            f"  {msg}: Reduction Percentages: Edges ({num_edges_start} -> {num_edges_end}): {per_edges * 100:.2f}%, Nodes ({num_nodes_start} -> {num_nodes_end}): {per_nodes * 100:.2f}%, Terminals ({num_terminals_start} -> {num_terminals_end}): {per_terminals * 100:.2f}%, Roots ({num_terminal_roots_start} -> {num_terminal_roots_end}): {per_roots * 100:.2f}% ({num_super_terminals} super terminals) [Trees solved: {self.solved_trees}]"
        )

    def critical_steiner_nodes(self):
        """Identifies all Steiner nodes that are critical to the graph."""
        nodes = set(self.graph.node_indices())
        non_steiner_nodes = set(self.non_steiner_nodes())
        steiner_nodes = nodes - non_steiner_nodes
        critical_nodes = set()
        for steiner_node in steiner_nodes:
            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_node(steiner_node)
            violates_demand = any(
                # Traverse from terminal to root to ensure super terminal reachability
                not rx.has_path(tmp, t, r)
                for r, terminals in self.terminal_sets.items()
                for t in terminals
            )
            if violates_demand:
                critical_nodes.add(steiner_node)
        critical_nodes = {self.get_node_key(n) for n in critical_nodes}
        return critical_nodes

    def get_node_key(self, node: int) -> int:
        """Logging helper."""
        key = self.node_to_key.get(node, None)
        if key is None:
            raise RuntimeError(f"Node {node} not found in index <-> key map.")
        return key

    # Operation helpers
    def get_terminal_set_roots(self, terminals: Iterable[int]) -> list[int]:
        return [next(r for r, ts in self.terminal_sets.items() if t in ts) for t in terminals]

    def non_steiner_nodes(self) -> set[int]:
        """Returns the set of nodes that are not steiner nodes in the graph."""
        all_active = set(self.terminal_sets.keys()) | set().union(*self.terminal_sets.values())

        if self.super_root_index is not None and self.super_root_index in self.terminal_sets:
            potential_roots = {i for i in self.potential_roots if self.graph.has_node(i)}
            return potential_roots | all_active

        return all_active

    def reduction_degree(self, node: int) -> int:
        """Returns the degree of the given node pertaining to reduction semantics."""
        # NOTE: In the node weighted setting all edges are treated as unweighted and undirected
        #       reresentated by bi-directional anti-parallel arcs with the exception of a super root.
        #       If present, the super root has incoming edges only from each potential root
        #       that can satisfy an associated super terminal demand.
        #       As such, for the purposes of graph reduction the SUPER_ROOT degree is zero,
        #       potential roots use in_degree and out_degree is the determinant.
        return min(self.graph.in_degree(node), self.graph.out_degree(node))

    def reduction_neighbors(self, node: int) -> set[int]:
        """Returns the set of neighbors of the given node that are used in the reduction semantics."""
        # NOTE: In the node weighted setting all edges are treated as unweighted and undirected
        #       reresentated by bi-directional anti-parallel arcs with the exception of a super root.
        #       If present, the super root has incoming edges only from each potential root
        #       that can satisfy an associated super terminal demand.
        #       As such, for the purposes of graph reduction neighbors can be considered in the undirected
        #       sense and taken from the predecessor set.
        return set(self.graph.predecessor_indices(node))

    def set_edge_weights(self):
        """Sets the weights of all edges to the weight of the destination node."""
        non_steiner_nodes = self.non_steiner_nodes()
        for u, v in self.graph.edge_list():
            weight = self.graph[v][PAYLOAD_WEIGHT_KEY]
            if v in non_steiner_nodes - self.potential_roots:
                weight = 0
            self.graph.update_edge(u, v, {PAYLOAD_WEIGHT_KEY: weight})

    def validate_reachability(self):
        logger.trace("validate_reachability...")

        all_ts_reachable = True
        for r, terminals in self.terminal_sets.items():
            for t in terminals:
                # NOTE: traverse from terminal to root to ensure super terminal reachability
                if not rx.has_path(self.graph, t, r):
                    logger.error(f"Unreachable ts pair: {self.get_node_key(t)} → {self.get_node_key(r)}")
                    all_ts_reachable = False
        if not all_ts_reachable:
            raise RuntimeError("Unreachable pairs")
        logger.trace("    ...successful")

    def choose_minimax_root(self, candidate_roots: set[int], terminals: set[int]) -> int:
        """
        Given a set of candidate roots and a merged terminal set,
        return the root r that minimizes max_t d(r, t).

        This is the minimax root selection used when trees collide.
        """
        best_root = next(iter(candidate_roots))
        best_value = float("inf")

        for r in candidate_roots:
            # Compute distances from r to all terminals
            path_lengths = [self.shortest_path_length(r, t) for t in terminals]

            # Compute the maximum distance to terminals
            max_rt = max(path_lengths)
            if max_rt < best_value:
                best_value = max_rt
                best_root = r

        return best_root

    def dw_remaining_complexity(self):
        complexity = 0  # 3^t * n + 2^t * n^2 for Dreyfus Wagner

        num_nodes = self.graph.num_nodes()
        num_roots = len(self.terminal_sets)
        num_terminals = len(set().union(*self.terminal_sets.values()))
        num_super_terminals = (
            len(self.terminal_sets[self.super_root_index]) if self.super_root_index is not None else 0
        )

        for choose_count in range(1, num_roots + 1):
            for comb in combinations(self.terminal_sets.keys(), choose_count):
                num_terms = sum(len(self.terminal_sets[c]) for c in comb) + choose_count
                complexity += (3**num_terms * num_nodes) + (2**num_terms) * (num_nodes**2)

        # TODO: Change back to info level after seeing how well scipstp scales...
        logger.warning(
            f"    remaining complexity: {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |n|: {num_nodes} (st: {num_super_terminals})"
        )

        return complexity

    def shortest_path_length(self, source_node, target_node) -> float:
        """Returns the length of the shortest path from source_node to target_node."""
        path_lengths = rx.dijkstra_shortest_path_lengths(
            self.graph,
            node=source_node,
            edge_cost_fn=lambda e: e[PAYLOAD_WEIGHT_KEY],  # Maps edge head weight
            goal=target_node,
        )
        length = path_lengths[target_node] if target_node in path_lengths else float("inf")
        return length

    def _get_maximum_rt_distance(self) -> float:
        """Calculates and returns the maximum root to terminal distance from shortest paths."""
        max_rt_distance = 1
        for r, terminals in self.terminal_sets.items():
            for t in terminals:
                if r == self.super_root_index:
                    # Ensure path goes from terminal to root to handle super-root reachability
                    max_rt_distance = max(max_rt_distance, self.shortest_path_length(t, r))
                else:
                    max_rt_distance = max(max_rt_distance, self.shortest_path_length(r, t))
        return max_rt_distance

    @property
    def min_max_rt_distance(self) -> float:
        """Updates and returns the maximum root to terminal distance."""
        self.set_edge_weights()
        max_rt_distance = min(self._global_min_max_drt, self._get_maximum_rt_distance())
        if max_rt_distance != self._global_min_max_drt and max_rt_distance > 0:
            logger.trace(f"  => max_rt_distance changed from {self._global_min_max_drt} to {max_rt_distance}")
            self._global_min_max_drt = int(max_rt_distance)
        logger.trace(f"  => max_rt_distance: {max_rt_distance}")

        return max_rt_distance

    @property
    def node_index_map(self) -> dict[int, int]:
        return {u: i for i, u in enumerate(self.graph.node_indices())}

    @property
    def node_weight_map(self) -> dict[int, int]:
        return {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

    # MARK: Primitive Terminal Set Actions

    def consume_terminal_set(self, terminal_sets: dict[int, set[int]], r_j: int, r_k: int):
        """𝓐ᵏ consumes 𝓐ʲ, 𝓐ᵏ survives containing all terminals of 𝓐ʲ and Ⓡʲ as a terminal.
        ** 𝓐ʲ ↣ 𝓐ᵏ **
        ```
        𝓐ᵏ ≔ 𝓐ᵏ ∪ 𝓐ʲ ∪ Ⓡʲ
        ```
        """
        if self.do_debug:
            assert r_j in terminal_sets, f"terminal set {self.get_node_key(r_j)} not in terminal sets"
            assert r_k in terminal_sets, f"terminal set {self.get_node_key(r_k)} not in terminal sets"
        terminal_sets[r_j].update(terminal_sets[r_k])
        terminal_sets[r_j].add(r_k)
        del terminal_sets[r_k]

    # MARK: Primitive Graph Actions

    def remove_node(self, u: int):
        """Removes the given node from the graph.
        `G ≔ G ∖ ⓤ`
        """
        if self.do_debug:
            assert self.graph.has_node(u), f"node {self.get_node_key(u)} not in graph"
        self.graph.remove_node(u)

    def remove_nodes_from(self, nodes: Iterable[int]):
        """Removes the given node from the graph.
        `∀ x : x ∈ nodes ⇒ G ≔ G ∖ ⓤ`
        """
        if self.do_debug:
            for x in nodes:
                assert self.graph.has_node(x), f"node {self.get_node_key(x)} not in graph"
        self.graph.remove_nodes_from(nodes)

    def move_edges(self, u: int, v: int):
        """Move all incident edges of ⓤ to ⓥ, avoiding self-loops and parallel edges, isolates ⓤ.
        ```
        ∀ x : (ⓤ ⇋ x ∈ G ∧ x ≠ ⓥ) ⇒ (ⓥ ⇋ x ∈ G), ∣ⓤ∣
        ```
        """
        if self.do_debug:
            assert self.graph.has_node(u), f"node {self.get_node_key(u)} not in graph"
            assert self.graph.has_node(v), f"node {self.get_node_key(v)} not in graph"

        # NOTE: We do not use self.reduction_neighbors here because a node can potentially have edges
        #       to the super root, and this just seems clearer...
        preds = list(self.graph.predecessor_indices(u))
        succs = list(self.graph.successor_indices(u))

        for w in preds:
            self.graph.remove_edge(w, u)
            if w != v and not self.graph.has_edge(w, v):
                self.graph.add_edge(w, v, {})

        for w in succs:
            self.graph.remove_edge(u, w)
            if w != v and not self.graph.has_edge(v, w):
                self.graph.add_edge(v, w, {})

    def absorb(self, u: int, v: int):
        """ⓥ absorbs ⓤ, ⓥ survives combining weights and embedded collapsed nodes.
        **ⓤ ⇴ ⓥ**
        ```
        move_edges(ⓤ, ⓥ)
        𝔀(ⓥ) ≔ 𝔀(ⓥ) + 𝔀(ⓤ)
        ⓥ(⦿) ≔ ⓥ(⦿) ⋃ ⓤ(⦿) ⋃ {ⓤ}
        remove(ⓤ)
        ```
        """
        self.move_edges(u, v)
        self.graph[v][PAYLOAD_WEIGHT_KEY] += self.graph[u][PAYLOAD_WEIGHT_KEY]
        self.graph[v][HYPERNODE_CONTENTS_KEY].update(self.graph[u][HYPERNODE_CONTENTS_KEY])
        self.graph[v][HYPERNODE_CONTENTS_KEY].add(u)
        self.remove_node(u)

    def consume(self, u: int, v: int):
        """ⓥ consumes ⓤ, ⓥ survives.
        **ⓤ ↣ ⓥ**
        ```
        𝓢 ≔ 𝓢 ⋃ ⓤ(⦿) ⋃ {ⓤ}
        move_edges(ⓤ, ⓥ)
        remove(ⓤ)
        ```
        """
        # Absorb ⓤ(⦿) into solution, ignoring the weight - since we are buying it.
        self.fixed_nodes.update(self.graph[u][HYPERNODE_CONTENTS_KEY])
        self.fixed_nodes.add(u)

        self.move_edges(u, v)
        self.remove_node(u)

    # MARK: Reductions

    def reduce_potential_roots(self):
        """
        Reduces potential root entries by distance.

        A potential root is only viable if it is within 2 * global_min_max_drt of any super terminal
        and is reachable by a super terminal. If not and if it has either an empty hyper contents
        or is a leaf it is removed from the set of potential roots.
        """
        # Nothing to do...
        if self.super_root_index is None:
            return

        # All super demands are satisfied. Just do cleanup...
        if not self.terminal_sets.get(self.super_root_index):
            self.terminal_sets.pop(self.super_root_index, None)
            self.graph.remove_node(self.super_root_index)
            self.super_root_index = None
            self.potential_roots = set()
            return

        logger.trace("reduce_potential_roots...")

        collisions: set[int] = set()
        guaranteed_sinks: set[int] = set()
        radius = 2 * self.min_max_rt_distance
        weight_map = self.node_weight_map

        for st in self.terminal_sets[self.super_root_index]:
            st_collisions = self._collision_envelope(st, weight_map, radius)
            collisions |= st_collisions

            # The collision envelope is a radius-based reachability guarantee, but
            # it does not guarantee a sink.
            # The one thing that's true unconditionally is that st is a surviving
            # super terminal, so it has a real path to sr so we need to guarantee
            # that a sink survives, independent of collision/radius outcome.
            if not st_collisions & self.potential_roots:
                paths = rx.all_shortest_paths(
                    self.graph, st, self.super_root_index, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY]
                )
                # The potential root will always be the node prior to the super root and if more
                # than one exists then there is still ambiguity regarding which is globally optimal
                # so we need to keep all of them.
                guaranteed_sinks |= {p[-2] for p in paths}

        num_old_roots = len(self.potential_roots)
        real_roots = set(self.terminal_sets.keys())
        self.potential_roots = (collisions & self.potential_roots) | real_roots | guaranteed_sinks

        if self.do_debug:
            logger.debug(f"  reduced potential roots from {num_old_roots} to {len(self.potential_roots)}")

    def reduce_demand_roots(self) -> int:
        """
        Reduces the number of distinct roots by merging fixed nodes (terminals or roots)
        that lie in the same connected component.

        Handles:
        > **`𝑻 ⭆ 𝑻', |𝑻'𐞪| ≤ |𝑻𐞪|, 𝐃 ≡ 𝐃'`**

        A subset of G consisting of the roots and terminals of all active demands with any
        connecting edges amoungst them is taken and augmented with direct edges between each
        terminal and its root. The resulting connected components then define the reduced
        terminal sets.

        Super Terminals are treated as independent of their root (since any super terminal
        can be connected to any potential root).

        This does not reduce the number of terminal-root pairs. It canonicalizes the root
        side of each pair and adds root->root pairs for collapsed roots which potentially
        increases the number of terminal-root pairs.
        """
        logger.trace("reduce_roots...")

        # NOTE: SUPER_ROOT directly connects to any base-town and any super terminal can
        #       be connected to any base-town, so it is excluded from the fixed set, and
        #       treated separately as isolated nodes during mapping.
        graph = self.graph
        super_root_index = self.super_root_index

        roots = set(self.terminal_sets.keys())
        num_roots = len(roots)
        roots.discard(self.super_root_index)
        tmp_fixed = roots | {t for ts in self.terminal_sets.values() for t in ts}

        self.set_edge_weights()

        # This acts as a union-find data structure for the terminal set clusters.
        sub = subgraph_stable(graph, tmp_fixed)
        for r, ts in self.terminal_sets.items():
            if r == self.super_root_index:
                continue
            for t in ts:
                if self.do_debug and not sub.has_node(r):
                    logger.error(
                        f"Terminal {self.get_node_key(t)} not connected to root {self.get_node_key(r)} missing {self.get_node_key(r)}:"
                    )
                if self.do_debug and not sub.has_node(t):
                    logger.error(
                        f"Root {self.get_node_key(r)} not connected to terminal {self.get_node_key(t)} missing {self.get_node_key(t)}:"
                    )
                sub.add_edge(r, t, {})
                sub.add_edge(t, r, {})
        components = rx.strongly_connected_components(sub)

        terminal_sets: dict[int, set[int]] = {}
        if super_root_index is not None:
            terminal_sets[super_root_index] = set()

        for comp in components:
            comp_roots = set(comp) & roots

            # Collect roots contained in component and select the one that minimizes the maximum d(root, terminal)
            # of the graph.
            rep = next(iter(comp_roots), None)
            if rep is not None:
                comp = set(comp)
                if len(comp_roots) > 1 and len(comp - comp_roots) > 0:
                    merged_terminals = comp - comp_roots
                    rep = self.choose_minimax_root(comp_roots, merged_terminals)
                terminal_sets[rep] = comp - {rep}
            elif super_root_index is not None:
                terminal_sets[super_root_index].update(comp)
            else:
                raise RuntimeError(
                    f"Component has no roots and super root is missing: {[self.get_node_key(n) for n in comp]}"
                )

        if self.super_root_index is not None and terminal_sets[self.super_root_index]:
            # Remove any terminal in the set that has a direct edge to super root
            # which happens when other reductions cascade into the super root.
            orig_set = set(terminal_sets[self.super_root_index])
            terminal_sets[self.super_root_index] = {
                t
                for t in terminal_sets[self.super_root_index]
                if not self.graph.has_edge(t, self.super_root_index)
            }
            for t in orig_set - terminal_sets[self.super_root_index]:
                # We need to ensure that any hyper node contents are added to fixed nodes
                # then remove this node from the terminal set but leave it available to any
                # other routing.
                self.fixed_nodes.update(self.graph[t][HYPERNODE_CONTENTS_KEY])
                self.fixed_nodes.add(t)
                self.graph[t][HYPERNODE_CONTENTS_KEY].clear()
                terminal_sets[self.super_root_index].discard(t)
                logger.trace(f"Removed direct connected super terminal: {self.get_node_key(t)}")

        if self.super_root_index is not None and not terminal_sets[self.super_root_index]:
            terminal_sets.pop(self.super_root_index)
            self.graph.remove_node(self.super_root_index)
            self.super_root_index = None
            self.potential_roots = set()

        self.terminal_sets = terminal_sets

        merges = num_roots - len(terminal_sets)

        if merges > 0:
            logger.info(f"  merged {merges} roots")
            if self.do_debug:
                self.dump_state("reduce_demand_roots", merges)
                self.validate_reachability()

        return merges

    def reduce_roots_via_articulation_points(self) -> int:
        """Merge roots whose connectivity is separated by cut vertices in the full graph."""
        logger.trace("reduce_via_articulation_points...")

        if self.graph.num_nodes() <= 2:
            return 0

        # --- Begin Undirected Graph ---

        # We only need subgraph for articulation point identification but we need to compress
        # the sparse indices before conversion to undirected.
        self.set_edge_weights()
        sub, node_map = self.graph.subgraph_with_nodemap(
            list(set(self.graph.node_indices()) - {self.super_root_index})
        )
        sub_undir = sub.to_undirected()

        articulations = rx.articulation_points(sub_undir)
        if not articulations:
            return 0

        # Convert back to original indices
        cuts = [node_map[a] for a in articulations]

        # --- End Undirected Graph ---

        merges = 0

        g_prime = self.graph.copy()
        active_roots = set(self.terminal_sets.keys())

        if self.super_root_index is not None:
            g_prime.remove_node(self.super_root_index)
            active_roots.discard(self.super_root_index)

        for cut in cuts:
            if cut == self.super_root_index:
                continue

            g_tmp = g_prime.copy()
            g_tmp.remove_node(cut)
            components = list(rx.strongly_connected_components(g_tmp))

            # Map each component to the set of active roots that have nodes in it
            comp_to_roots: dict[int, set[int]] = {}
            for comp_id, comp_nodes in enumerate(components):
                comp_set = set(comp_nodes)
                for r in active_roots:
                    # A root "touches" this component if root or any terminal is here
                    touches = comp_set & (self.terminal_sets[r] | {r})
                    if touches:
                        comp_to_roots.setdefault(comp_id, set()).add(r)

            # Find groups of roots that are split across multiple components
            root_to_comps: dict[int, set[int]] = {}
            for comp_id, roots_in_comp in comp_to_roots.items():
                for r in roots_in_comp:
                    root_to_comps.setdefault(r, set()).add(comp_id)

            # These trees _must_ collide; meaning we can merge them.
            multi_comp_roots = {r for r, comps in root_to_comps.items() if len(comps) > 1}
            if len(multi_comp_roots) < 2:
                continue

            # Merge them
            to_merge = list(multi_comp_roots)

            merged_ts = set()
            for r in to_merge:
                merged_ts.update(self.terminal_sets[r])

            to_merge = set(to_merge)
            if to_merge:
                rep = self.choose_minimax_root(to_merge, merged_ts)

                # The merged roots become terminals in the new cluster.
                self.terminal_sets[rep] = merged_ts
                self.terminal_sets[rep].update(to_merge)
                self.terminal_sets[rep].discard(rep)

                # Clear out the old roots
                for r in to_merge - {rep}:
                    self.terminal_sets.pop(r, None)
                    active_roots.discard(r)

                if rep in self.terminal_sets[rep]:
                    logger.error(f"Duplicate rep {rep} root: {self.get_node_key(rep)}")

                merges += len(to_merge) - 1

        if merges > 0:
            logger.info(f"  reduced {merges} bottleneck merged roots")
            if self.do_debug:
                self.dump_state("reduce_roots_via_articulation_points", merges)
                self.validate_reachability()

        return merges

    def reduce_adjacent_terminals(self) -> int:
        """Reduces the number of distinct terminals by collapsing adjacent terminals per root.

        NOTE: Terminal_sets must be disjoint by root, which is guaranteed by reduce_demand_roots.

        Handles:
        > **`⁝Ⓣʲ¹ ⇋ Ⓣʲ²⁝  ⭆  ⁝Ⓣʲ¹⁝, 𝐆 ∖ Ⓣʲ², 𝓢 ⋃ Ⓣʲ²(⦿)}`**
        > **`⁝Ⓡʲ ⇋ Ⓣʲ⁝  ⭆  ⁝Ⓡʲ⁝, 𝐆 ∖ Ⓣʲ, 𝓢 ⋃ Ⓡʲ(⦿)`**
        """
        logger.trace("reduce_adjacent_terminals...")

        def consume_adjacent_terminals(survivor: int, terminals: set[int]) -> bool:
            # NOTE: rustworkx adjacency queries of node indices not present in the graph return empty sets.
            neighbors = self.reduction_neighbors(survivor)
            consumables = neighbors & terminals
            for c in list(consumables):
                self.consume(c, survivor)
                terminals.discard(c)
            return bool(consumables)

        reduced_terminal_sets: dict[int, set[int]] = {r: set(ts) for r, ts in self.terminal_sets.items()}

        num_terminals = sum(len(ts) for ts in reduced_terminal_sets.values()) + len(reduced_terminal_sets)

        for r, terminals in reduced_terminal_sets.items():
            # Collapse terminal adjacent terminals
            made_progress = True
            while made_progress:
                made_progress = False
                for t in list(terminals):
                    made_progress |= consume_adjacent_terminals(t, terminals)

            # Collapse root adjacent terminals
            if r != self.super_root_index:
                consume_adjacent_terminals(r, terminals)

        # Drop empty sets
        self.terminal_sets = {r: ts for r, ts in reduced_terminal_sets.items() if ts}

        merges = num_terminals - (
            sum(len(ts) for ts in self.terminal_sets.values()) + len(self.terminal_sets)
        )

        if merges > 0:
            logger.info(f"  merged {merges} adjacent terminals")
            if self.do_debug:
                self.dump_state("reduce_adjacent_terminals", merges)
                self.validate_reachability()

        return merges

    def reduce_degree1_steiner_nodes(self) -> int:
        """Multi-pass degree-1 Steiner node reduction.

        If a Steiner node is a leaf, it cannot satisfy or bridge a demand.

        Handles:
        > **`|Ⓢ ⇋ ○⁝  ⭆  ○⁝, 𝐆 ∖ Ⓢ`**
        """
        logger.trace("reduce_degree1_steiner_nodes...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removables = []
        while deg1_nodes := {i for i in self.graph.node_indices() if self.reduction_degree(i) == 1}:
            deg1_nodes.difference_update(non_steiners)
            if not deg1_nodes:
                break
            self.graph.remove_nodes_from(deg1_nodes)
            removables.extend(deg1_nodes)

        removals = len(removables)
        if removals > 0:
            logger.info(f"  removed {removals} degree-1 steiner nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("reduce_degree1_steiner_nodes", removals)
                self.validate_reachability()

        return removals

    def reduce_degree1_terminals(self) -> int:
        """Single-pass degree-1 terminal reduction.

        If a terminal node is a leaf it must use its neighbor to satisfy a demand.

        Handles:
        > **`|Ⓣ ⇋ Ⓢ⁝  ⭆  Ⓣ ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**
        > **`|𝕥 ⇋ Ⓢ⁝  ⭆  𝕥 ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**

        """
        logger.trace("reduce_degree1_terminals...")

        terminal_sets = self.terminal_sets
        removals = 0

        for ts in terminal_sets.values():
            for t in list(ts):
                if self.reduction_degree(t) != 1:
                    continue

                succs = list(self.graph.successor_indices(t))
                s = succs[0]

                # Don't consume fixed nodes or SUPER_ROOT
                # (adjacent terminals are handled in reduce_adjacent_terminals)
                if s in self.fixed_nodes or s == self.super_root_index:
                    if self.do_debug:
                        roots = set(terminal_sets.keys())
                        logger.debug(
                            f"  ...skipping degree-1 reduction: Root: {s in roots} "
                            f"Super Root: {s == self.super_root_index} "
                            f"Adjacent Terminal: {s in ts} Fixed: {s in self.fixed_nodes}"
                        )
                    continue

                self.consume(s, t)
                removals += 1

        if removals > 0:
            logger.info(f"  removed {removals} degree-1 terminals")
            if self.do_debug:
                self.dump_state("reduce_degree1_terminals", removals)
                self.validate_reachability()

        return removals

    def reduce_degree1_roots(self) -> int:
        """Single-pass degree-1 root reduction.

        If a root node with unsatisfied demand is a leaf it must use its neighbor to satisfy a demand.

        Handles:
        > **`|Ⓡ ⇋ Ⓢ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓢ, 𝓢 ⋃ {Ⓢ, Ⓢ(⦿)}`**
        > **`|Ⓡᵏ ⇋ Ⓣᵏ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓣᵏ, 𝓢 ⋃ Ⓣᵏ(⦿), 𝓐ᵏ ∖ Ⓣᵏ}`**
        > **`|Ⓡᵏ ⇋ Ⓣʲ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ Ⓣʲ, 𝓢 ⋃ Ⓣʲ(⦿), 𝓐ᵏ ⋃ {𝓐ʲ ∖ Ⓣʲ}`**
        > **`|Ⓡᵏ ⇋ 𝕥ʲ⁝  ⭆  Ⓡ ⇋ ⁝, 𝐆 ∖ 𝕥ʲ, 𝓢 ⋃ 𝕥ʲ(⦿), 𝕊 ∖ 𝕥`**
        """
        logger.trace("reduce_degree1_roots...")

        # NOTE: Potential roots are not considered as roots here because a potential
        #       root absorbs rather than consumes its neighbors and can only absorb
        #       a 2-degree neighbor since if it consumes a neighbor with deg() > 2
        #       it can end up in the optimal path of a different demand set which
        #       would incur the cost of the previously absorbed neighbors in its
        #       hyper node contents which may not be part of the optimal solution.

        terminal_sets = self.terminal_sets

        roots = set(terminal_sets.keys())
        roots.discard(self.super_root_index)
        # Only active roots are considered
        roots = roots - {r for r, ts in terminal_sets.items() if not ts}
        all_terminals = set(terminal_sets.keys()) | roots

        removables = []

        for r_k in list(roots):
            # NOTE: If r_k is the root of a previously consumed terminal set then it is now a terminal.
            if r_k not in terminal_sets:
                continue

            # A root, in the presence of a super_root, can have 2 successors
            # but only ever 1 predecessor when reduction_degree == 1
            if self.reduction_degree(r_k) != 1:
                continue
            neighbor = next(iter(self.reduction_neighbors(r_k)))
            if neighbor in all_terminals:
                # Don't consume adjacent roots or terminals, they are handled during adjacent root reduction
                continue

            self.consume(neighbor, r_k)
            removables.append(neighbor)

            r_j = next((r for r, ts in terminal_sets.items() if neighbor in ts), None)
            if r_j is not None:
                # The neighbor was not a Steiner node, fix terminal sets...
                terminal_sets[r_j].remove(neighbor)
                if r_k != r_j and r_j != self.super_root_index:
                    self.consume_terminal_set(terminal_sets, r_j, r_k)

        removals = len(removables)
        if removals > 0:
            # Drop empty sets
            self.terminal_sets = {r: ts for r, ts in terminal_sets.items() if ts}
            logger.info(f"  removed {removals} degree-1 root Steiner nodes")
            if self.do_debug:
                logger.trace(f"    {[self.get_node_key(i) for i in removables]}")
                self.dump_state("reduce_degree1_roots", removals)
                self.validate_reachability()

        return removals

    def merge_adjacent_degree2_steiner_chains(self) -> int:
        """Merge degree-2 Steiner chains that are adjacent to each other.

        If a degree-2 Steiner node is adjacent to another degree-2 Steiner node they
        can only satisfy a demand by bridging both Steiner nodes, so they can be
        merged by absorption.

        Handles:
        > **`⁝⇋ Ⓢ¹ ⇋ Ⓢ² ⇋⁝  ⭆  ⁝⇋ Ⓢ¹ ⇋⁝, Ⓢ² ⇴ Ⓢ¹`**

        """
        logger.trace("merge_adjacent_degree2_steiner_chains...")

        non_steiners = self.non_steiner_nodes()
        removables = []

        made_progress = True
        while made_progress:
            made_progress = False

            for node in set(self.graph.node_indices()) - non_steiners:
                # A stale entry will have reduction_degree == 0
                if self.reduction_degree(node) != 2:
                    continue

                neighbor = next(
                    (
                        n
                        for n in self.reduction_neighbors(node)
                        if self.reduction_degree(n) == 2 and n not in non_steiners
                    ),
                    None,
                )
                if neighbor:
                    self.absorb(neighbor, node)
                    # logger.trace(f"    {self.get_node_key(neighbor)} ⇴ {self.get_node_key(node)}")
                    removables.append(neighbor)
                    made_progress = True
                    break

        removals = len(removables)
        if removals > 0:
            logger.info(f"  absorbed {removals} degree-2 Steiner chain nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("merge_adjacent_degree2_steiner_chains", removals)
                self.validate_reachability()

        return removals

    def reduce_degree2_steiner_dominance(self) -> int:
        """
        Removes Steiner nodes of degree 2 that are dominated by alternate paths.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝ ⭆  ⁝ⓤ ⇋ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔ 𝔀(ⓤ ↠∣Ⓢ∣↠ ⓥ) <= 𝔀(ⓤ ⇋ Ⓢ ⇋ ⓥ)`**
        """
        logger.trace("reduce_degree2_steiner_dominance...")

        graph = self.graph
        non_steiner_nodes = self.non_steiner_nodes()
        removables = []

        made_progress = True
        while made_progress:
            made_progress = False

            # Each iteration recomputes the edge weights and steiner nodes...
            self.set_edge_weights()
            for node in set(graph.node_indices()) - non_steiner_nodes:
                if self.reduction_degree(node) != 2:
                    continue
                neighbors = list(self.reduction_neighbors(node))
                u, v = neighbors
                if u == v:
                    continue

                try:
                    paths = rx.all_shortest_paths(graph, u, v, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
                except rx.NoPathFound:
                    continue

                dominated = any(node not in P for P in paths)
                if not dominated:
                    continue

                self.remove_node(node)

                removables.append(node)
                made_progress = True

                break

        removals = len(removables)
        if removals > 0:
            logger.info(f"  removed {removals} degree-2 dominated nodes")
            if self.do_debug:
                logger.trace(f"    {[self.get_node_key(i) for i in removables]}")
                self.dump_state("reduce_degree2_steiner_dominance", removals)
                self.validate_reachability()

        return removals

    def reduce_degree2_articulation(self) -> int:
        """
        Processes degree-2 articulation point / 2-bridge nodes for inclusion/exclusion.

        If an articulation node's removal violates any demands it must be included in the solution, if
        its removal does not violate any demands and no super terminal would prefer it in an optimal
        solution then it is safe to be excluded from the solution.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ∣ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔  Ⓡ (↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃`**
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ⇋ ⓥ⁝, ⓤ ↣ ○  ⇔  Ⓡ ¬(↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃, dist(𝕥, 𝕡) <= dist(𝕥, Ⓢ)`**
        """
        logger.trace("reduce_degree2_articulation...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        fixes = 0
        removals = 0

        def is_potential_violation(u: int) -> bool:
            from collections import deque

            if self.super_root_index is None:
                return False

            for t in self.terminal_sets[self.super_root_index]:
                queue: deque[int] = deque([t])
                visited = {t}
                while queue:
                    curr = queue.popleft()
                    if curr == u:
                        return True
                    if curr in non_steiners:
                        break

                    for neigh in self.graph.successor_indices(curr):
                        if neigh not in visited:
                            visited.add(neigh)
                            queue.append(neigh)

            return False

        # Get articulation points
        tmp_undir = self.graph.to_undirected()
        if self.super_root_index is not None:
            tmp_undir.remove_node(self.super_root_index)
        articulations = rx.articulation_points(tmp_undir)
        if not articulations:
            return 0

        # Map back to self.graph indices
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}
        articulations = {tmp_map[i] for i in articulations}

        articulations = {u for u in articulations if u not in non_steiners and self.reduction_degree(u) == 2}

        for u in articulations:
            if not self.graph.has_node(u):
                continue

            neighbors = list(self.graph.successor_indices(u))
            if len(neighbors) != 2:
                continue
            a, b = neighbors

            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_node(u)

            violates_demand = any(
                not rx.has_path(tmp, r, t) for r, terminals in self.terminal_sets.items() for t in terminals
            )

            if violates_demand:
                # Must include
                self.consume(u, a if a in self.fixed_nodes else b)
                fixes += 1
            elif not is_potential_violation(u):
                # Safe to remove
                self.remove_node(u)
                removals += 1
            # else:
            #     # Can not certify because of ambiguity with super terminal optimality
            #     pass

        if removals > 0 or fixes > 0:
            logger.info(f"  fixed {fixes} degree-2 bridges, removed {removals} redundant degree-2 nodes")
            if self.do_debug:
                self.dump_state("reduce_degree2_articulation", removals + fixes)
                self.validate_reachability()

        return removals + fixes

    def reduce_degree2_articulation_super(self) -> int:
        """
        Processes degree-2 articulation point / 2-bridge nodes for inclusion/exclusion.

        If an articulation node's removal violates any demands it must be included in the solution, if
        its removal does not violate any demands and no super terminal would prefer it in an optimal
        solution then it is safe to be excluded from the solution.

        Handles:
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ∣ ⓥ⁝, 𝐆 ∖ Ⓢ  ⇔  Ⓡ (↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃`**
        > **`⁝ⓤ ⇋ Ⓢ ⇋ ⓥ⁝  ⭆  ⁝ⓤ ⇋ ⓥ⁝, ⓤ ↣ ○  ⇔  Ⓡ ¬(↠∣Ⓢ∣↠) Ⓣ ∀ 𝐃, dist(𝕥, 𝕡) <= dist(𝕥, Ⓢ)`**
        """
        logger.trace("reduce_degree2_articulation...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        fixes = 0
        removals = 0

        # Get articulation points
        tmpG = self.graph.copy()
        if self.super_root_index is not None:
            tmpG.remove_node(self.super_root_index)
        tmp_undir = tmpG.to_undirected(multigraph=False)
        articulations = rx.articulation_points(tmp_undir)
        if not articulations:
            return 0

        # Map back to self.graph indices
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}
        articulations = {tmp_map[i] for i in articulations}

        articulations = {u for u in articulations if u not in non_steiners and self.reduction_degree(u) == 2}
        if not articulations:
            return 0

        # Compute collision envelopes
        sr_index = self.super_root_index
        st_collisions = {}
        weight_map = self.node_weight_map
        if sr_index is not None:
            for st in self.terminal_sets[sr_index]:
                st_collisions[st] = self._collision_envelope(st, weight_map, 2 * self.min_max_rt_distance)

        for u in articulations:
            if not self.graph.has_node(u):
                continue

            neighbors = list(self.graph.successor_indices(u))
            if len(neighbors) != 2:
                continue
            a, b = neighbors

            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_node(u)

            # NOTE: Traverse from terminal to root (for super-root presence)
            violates_demand = any(
                not rx.has_path(tmp, t, r) for r, terminals in self.terminal_sets.items() for t in terminals
            )

            if violates_demand:
                # Must include
                self.consume(u, a if a in self.fixed_nodes else b)
                fixes += 1
                continue

            if sr_index is None:
                continue

            # Super terminal violation test
            components = rx.strongly_connected_components(tmp)
            potential_violation = False
            for st in self.terminal_sets[sr_index]:
                for c in components:
                    if st in c and not st_collisions[st].issubset(set(c)):
                        potential_violation = True
                        break
                if potential_violation:
                    break

            if not potential_violation:
                # Safe to remove
                self.remove_node(u)
                removals += 1
            # else:
            #     # Can not certify because of ambiguity with super terminal optimality
            #     pass

        if removals > 0 or fixes > 0:
            logger.info(f"  fixed {fixes} degree-2 bridges, removed {removals} redundant degree-2 nodes")
            if self.do_debug:
                self.dump_state("reduce_degree2_articulation", removals + fixes)
                self.validate_reachability()

        return removals + fixes

    def reduce_steiner_bridges(self) -> int:
        """Bridge reduction for Steiner Forest."""
        logger.trace("reduce_steiner_bridges...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removals = 0
        edge_removals = 0

        # Get bridges
        tmp_undir = self.graph.to_undirected(multigraph=False)
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}

        if self.super_root_index is not None:
            undir_super_root_index = next(iter([i for i, j in tmp_map.items() if j == self.super_root_index]))
            tmp_undir.remove_node(undir_super_root_index)

        bridges = rx.bridges(tmp_undir)
        if not bridges:
            return 0

        bridges = sorted((tmp_map[min(u, v)], tmp_map[max(u, v)]) for u, v in bridges)  # ty:ignore[invalid-assignment]
        logger.trace(f"  {bridges=}")

        for u, v in bridges:
            if not (self.graph.has_node(u) and self.graph.has_node(v)):
                continue

            deg_u = self.reduction_degree(u)
            deg_v = self.reduction_degree(v)
            if not ((deg_u >= 2 and deg_v > 2) or (deg_v >= 2 and deg_u > 2)):
                continue

            steiner_u = u not in non_steiners
            steiner_v = v not in non_steiners
            if not (steiner_u or steiner_v):
                continue

            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_edge(u, v)
            tmp.remove_edge(v, u)

            # Since super root was removed any super terminal test would violate which
            # could potentially cause extra edges to be forced in the solution...
            violates_demand = any(
                not rx.has_path(tmp, t, r)
                for r, terminals in self.terminal_sets.items()
                for t in terminals
                if r != self.super_root_index
            )

            if not violates_demand:
                # In the presence of super root we can't certify exclusion without testing nearby super terminals
                # or the presence of a potential root in the component and since hanging clusters consolidate
                # towards roots/potential roots that would almost always be the case. So, we just skip for
                # now instead of incurring the ovehead of super terminal testing.
                if self.super_root_index is not None:
                    continue

                # NOTE: We can't certify exclusion of either endpoint of a single bridge because it says nothing
                #       about the two endpoints in an optimal Solution. And in a node weighted problem if both
                #       end up being selected then the edge is naturally selected.
                #       But, if the created component from removing the bridge contains no terminals or roots
                #       or only one arborescence then it is safe to remove the bridge edge.
                #       In that case we can remove the edge but not consume either endpoint.
                sccs = rx.strongly_connected_components(tmp)
                for scc in sccs:
                    set_scc = set(scc)

                    # If the component contains no fixed nodes then it is safe to remove
                    if not set_scc & self.fixed_nodes:
                        # The component contains no fixed nodes, thus no-demand and no structure violations
                        self.graph.remove_edge(u, v)
                        self.graph.remove_edge(v, u)
                        edge_removals += 1
                        logger.trace(
                            f"  removed bridge edge: no fixed nodes: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                        )
                        break

                    scc_roots = set_scc & set(self.terminal_sets.keys())
                    if len(scc_roots) == 1:
                        # A single root is present, so if all of its terminals are also present then it is safe to remove
                        scc_root = next(iter(scc_roots))
                        if set_scc & self.terminal_sets[scc_root]:
                            self.graph.remove_edge(u, v)
                            self.graph.remove_edge(v, u)
                            edge_removals += 1
                            logger.trace(
                                f"  removed bridge edge: single root: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                            )
                            break
                continue

            # The edge must be traversed in an optimal Solution...
            # Consume Steiner into the other side
            if steiner_u:
                self.consume(u, v)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(u)}")
            else:
                self.consume(v, u)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(v)}")
            removals += 1

        if removals > 0:
            logger.info(
                f"  consumed {removals} Steiner bridge nodes and removed {edge_removals} bridge edges"
            )
            if self.do_debug:
                self.dump_state("reduce_bridge_steiner_consumption", removals + edge_removals)
                self.validate_reachability()

        return removals

    def reduce_steiner_bridges_super(self) -> int:
        """Bridge reduction for Steiner Forest."""
        logger.trace("reduce_steiner_bridges...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        removals = 0
        edge_removals = 0

        # Get bridges
        tmpG = self.graph.copy()
        sr_index = self.super_root_index
        if sr_index is not None:
            tmpG.remove_node(sr_index)

        tmp_undir = tmpG.to_undirected(multigraph=False)
        tmp_map = {i: u for i, u in enumerate(self.graph.node_indices())}

        bridges = rx.bridges(tmp_undir)
        if not bridges:
            return 0

        bridges = sorted((tmp_map[min(u, v)], tmp_map[max(u, v)]) for u, v in bridges)  # ty:ignore[invalid-assignment]
        logger.trace(f"  {bridges=}")

        # Precompute super terminal collision envelopes once, up front -- same construction
        # and staleness caveat as reduce_degree2_articulation: computed against the graph
        # state at loop start, so a removal earlier in this same pass can only make a later
        # check MORE conservative (a stale, wider envelope is harder to satisfy issubset
        # against), never unsound. Self-corrects on the pipeline's next iteration.
        st_collisions: dict[int, set[int]] = {}
        if sr_index is not None:
            weight_map = self.node_weight_map
            radius_cap = 2 * self.min_max_rt_distance
            for st in self.terminal_sets[sr_index]:
                st_collisions[st] = self._collision_envelope(st, weight_map, radius_cap)

        for u, v in bridges:
            if not (self.graph.has_node(u) and self.graph.has_node(v)):
                continue

            deg_u = self.reduction_degree(u)
            deg_v = self.reduction_degree(v)
            if not ((deg_u >= 2 and deg_v > 2) or (deg_v >= 2 and deg_u > 2)):
                continue

            steiner_u = u not in non_steiners
            steiner_v = v not in non_steiners
            if not (steiner_u or steiner_v):
                continue

            # Demand separation test
            tmp = self.graph.copy()
            tmp.remove_edge(u, v)
            tmp.remove_edge(v, u)

            violates_demand = any(
                not rx.has_path(tmp, t, r)
                for r, terminals in self.terminal_sets.items()
                for t in terminals
                if r != self.super_root_index
            )

            if not violates_demand:
                sccs = list(rx.strongly_connected_components(tmp))

                if sr_index is not None:
                    # Super terminal collision certification. If every super terminal's full
                    # collision envelope stays inside its own resulting component, this edge
                    # was never on a route any optimal solution needed -- safe to consider for
                    # removal. If any envelope now spans the cut, we can't certify: skip.
                    scc_of: dict[int, int] = {}
                    for scc_i, scc in enumerate(sccs):
                        for n in scc:
                            scc_of[n] = scc_i

                    violates_st = False
                    for st, envelope in st_collisions.items():
                        scc_i = scc_of.get(st)
                        if scc_i is None:
                            continue  # resolved by an earlier iteration of this same pass
                        if not envelope.issubset(sccs[scc_i]):
                            violates_st = True
                            break

                    if violates_st:
                        continue

                    # sr_index's own SCC is always a content-free trivial singleton (pure sink,
                    # never in fixed_nodes) -- it would vacuously satisfy "no fixed nodes" below
                    # regardless of what the real split looks like, so it must not be considered
                    # as one of the candidate pieces.
                    sccs = [scc for scc in sccs if sr_index not in scc]

                # NOTE: We can't certify exclusion of either endpoint of a single bridge because it says nothing
                #       about the two endpoints in an optimal Solution. And in a node weighted problem if both
                #       end up being selected then the edge is naturally selected.
                #       But, if the created component from removing the bridge contains no terminals or roots
                #       or only one arborescence then it is safe to remove the bridge edge.
                #       In that case we can remove the edge but not consume either endpoint.
                for scc in sccs:
                    set_scc = set(scc)

                    if not set_scc & self.fixed_nodes:
                        self.graph.remove_edge(u, v)
                        self.graph.remove_edge(v, u)
                        edge_removals += 1
                        logger.trace(
                            f"  removed bridge edge: no fixed nodes: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                        )
                        break

                    scc_roots = set_scc & set(self.terminal_sets.keys())
                    if len(scc_roots) == 1:
                        scc_root = next(iter(scc_roots))
                        if set_scc & self.terminal_sets[scc_root]:
                            self.graph.remove_edge(u, v)
                            self.graph.remove_edge(v, u)
                            edge_removals += 1
                            logger.trace(
                                f"  removed bridge edge: single root: ({self.get_node_key(u)}, {self.get_node_key(v)})"
                            )
                            break
                continue

            # The edge must be traversed in an optimal Solution...
            if steiner_u:
                self.consume(u, v)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(u)}")
            else:
                self.consume(v, u)
                logger.trace(f"  consumed Steiner bridge node: {self.get_node_key(v)}")
            removals += 1

        if removals > 0 or edge_removals > 0:
            logger.info(
                f"  consumed {removals} Steiner bridge nodes and removed {edge_removals} bridge edges"
            )
            if self.do_debug:
                self.dump_state("reduce_bridge_steiner_consumption", removals + edge_removals)
                self.validate_reachability()

        return removals

    def reduce_degreek_steiner_dominance(self, max_degree: int = 4) -> int:
        """
        Removes Steiner nodes of small degree k that are dominated by alternate paths.
        Strict dominance is used in the presence of fixed nodes.

        Reduction Rule:
        ```
        Ⓢ : N(Ⓢ) = {○₁, ○₂, …, ○ₖ}, 2 < k ≤ K
        ∧ ∀ {○ᵢ, ○ⱼ} ⊆ N(Ⓢ):
            (○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) exists
            if N(Ⓢ) ∩ 𝓢:
                ∧ 𝔀(○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) < 𝔀(○ᵢ 🡒 Ⓢ 🡒 ○ⱼ)
            else:
                ∧ 𝔀(○ᵢ ↠∣Ⓢ∣↠ ○ⱼ) ≤ 𝔀(○ᵢ 🡒 Ⓢ 🡒 ○ⱼ)

        ⭆  𝐆 ≔ 𝐆 ∖ Ⓢ
        ```
        """

        logger.trace("reduce_degreek_steiner_dominance...")

        self.set_edge_weights()
        graph = self.graph
        non_steiners = self.non_steiner_nodes()
        removals = 0

        def dijkstra_avoiding(start: int, forbidden: int) -> dict[int, float]:
            """Dijkstra that skips a single forbidden node."""

            dist: dict[int, float] = {start: 0.0}
            heap = [(0.0, start)]

            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue

                for v in graph.successor_indices(u):
                    if v == forbidden:
                        continue

                    w = graph.get_edge_data(u, v)[PAYLOAD_WEIGHT_KEY]
                    nd = d + w

                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))

            return dist

        made_progress = True
        while made_progress:
            made_progress = False

            for s in set(graph.node_indices()) - non_steiners:
                deg = graph.out_degree(s)

                if deg <= 2 or deg > max_degree:
                    continue

                neighbors = list(graph.successor_indices(s))
                if len(neighbors) != deg:
                    continue

                dominated = True
                has_fixed_neighbor = set(neighbors) & self.fixed_nodes

                for i in range(len(neighbors)):
                    if not dominated:
                        break

                    u = neighbors[i]

                    # Full distances
                    try:
                        dist_full = rx.dijkstra_shortest_path_lengths(
                            graph, u, lambda e: e[PAYLOAD_WEIGHT_KEY]
                        )
                    except Exception:  # noqa: BLE001
                        dominated = False
                        break

                    # Distances avoiding s
                    dist_avoid = dijkstra_avoiding(u, s)

                    for j in range(i + 1, len(neighbors)):
                        v = neighbors[j]

                        if v not in dist_full or v not in dist_avoid:
                            dominated = False
                            break

                        if dist_avoid[v] > dist_full[v] or (
                            dist_avoid[v] == dist_full[v] and has_fixed_neighbor
                        ):
                            dominated = False
                            break

                if not dominated:
                    continue

                self.remove_node(s)
                made_progress = True

                removals += 1
                break

        if removals > 0:
            logger.info(f"  removed {removals} degree-k dominated nodes")
            if self.do_debug:
                self.dump_state("reduce_degreek_steiner_dominance", removals)
                self.validate_reachability()

        return removals

    def reduce_steiner_triangle_degree2_legs(self):
        """Absorbs degree-2 Steiner nodes adjacent to degree-3 Steiner triangles.

        Handles:
        > **`⁝⋯Ⓢᵃ⇋Ⓢᵘ⇋Ⓢᵇ⋯⁝ ∧ ⋯Ⓢᵃ⇋Ⓢᵇ⋯ ∧ ⋯Ⓢᵘ⇋Ⓢᵛ⇋⁝  ⭆  ⓥ ⇴ ⓤ`**
        """
        logger.trace("reduce_degree3_steiner_triangle_legs...")

        non_steiners = self.non_steiner_nodes()
        removals = 0

        # Single pass iteration...
        d3_steiners = {
            v for v in set(self.graph.node_indices()) - non_steiners if self.reduction_degree(v) == 3
        }
        for node in d3_steiners:
            if self.reduction_degree(node) != 3:
                continue
            neighbors = self.reduction_neighbors(node)
            if neighbors & non_steiners:
                continue

            # One must be degree-2 and the other two must be adjacent to each-other (for the triangle)
            degree2_neighbor = next((n for n in neighbors if self.reduction_degree(n) == 2), None)
            if degree2_neighbor is None:
                continue

            neighbors = neighbors - {degree2_neighbor}
            a, b = neighbors
            if not self.graph.has_edge(a, b):
                continue

            # It is safe to absorb the degree-2 node into the degree-3 triangle
            self.absorb(degree2_neighbor, node)
            removals += 1

        if removals > 0:
            logger.info(f"  removed {removals} degree-3 steiner triangle legs")
            if self.do_debug:
                self.dump_state("reduce_degree3_steiner_triangle_legs", removals)
                self.validate_reachability()

        return removals

    def reduce_blocked_roots(self):
        """Merge roots whose arborescences block another root's demand.

        Handles:
        > **` ⁝ Ⓣʲ ¬(↠∣𝓐ᵏ∣↠) Ⓡʲ ⁝  ⭆  𝓐ʲ ↣ 𝓐ᵏ`**
        """
        logger.trace("reduce_blocked_roots...")

        def is_blocking_cluster(t: int, r: int, blockers: set[int]) -> bool:
            visited = set()
            stack = [t]

            while stack:
                u = stack.pop()

                if u == r:
                    return False

                if u in visited or u in blockers:
                    continue

                visited.add(u)

                # NOTE: We do not use self.reduction_neighbors here because the super root could potentially
                #       be blocked by another root, and this allows for super-terminal to regular root merges
                stack.extend(self.graph.successor_indices(u))
                stack.extend(self.graph.predecessor_indices(u))

            return True

        terminal_sets = self.terminal_sets
        merges = 0
        made_progress = True

        while made_progress:
            made_progress = False

            # 𝓐𐞪 lookup
            # Blocker-first ordering: larger clusters first
            cluster_nodes = {r: set(ts) | {r} for r, ts in terminal_sets.items()}
            sorted_roots = sorted(terminal_sets.keys(), key=lambda r: len(terminal_sets[r]), reverse=True)

            # Try each root as a potential blocker
            for r_k in sorted_roots:
                if r_k == self.super_root_index:
                    continue

                Ck = cluster_nodes[r_k]

                # Test whether r_j blocks any other root r_k
                for r_j, terminals in terminal_sets.items():
                    if r_j == r_k or r_j == self.super_root_index:
                        continue

                    if any(is_blocking_cluster(t, r_j, Ck) for t in terminals):
                        # Perform merge: 𝓐ʲ ↣ 𝓐ᵏ
                        self.consume_terminal_set(terminal_sets, r_j, r_k)
                        merges += 1
                        made_progress = True
                        break

                if made_progress:
                    break

        # Drop empty sets
        self.terminal_sets = {r: ts for r, ts in terminal_sets.items() if ts}

        if merges > 0:
            logger.info(f"  merged {merges} blocked roots")
            if self.do_debug:
                self.dump_state("reduce_blocked_roots", merges)
                self.validate_reachability()

        return merges

    def reduce_degreek_enclosed_steiner(self) -> int:
        """
        General reduction for enclosed Steiner nodes surrounded by
        degree-2 terminals of the same arborescence.

        If there exists at least one unambiguous outer Steiner endpoint u*
        such that w(e) <= w(u*), then e must be purchased in any optimal
        solution (even if all other outer endpoints are "free" due to
        external sharing).

        This generalizes both the degree-2 and degree-3 enclosed cases.
        """
        logger.trace("reduce_enclosed_same_root_steiner...")

        non_steiners = self.non_steiner_nodes()
        non_steiners.discard(self.super_root_index)

        def is_unambiguous_outer_endpoint(u: int, t: int) -> bool:
            if u in non_steiners or u == self.super_root_index:
                return False

            deg = self.reduction_degree(u)
            if deg == 2:
                return True
            if deg != 3:
                return False

            # Degree 3 triangle case
            neighbors = self.reduction_neighbors(u)
            neighbors.discard(t)

            if len(neighbors) != 2:
                return False

            x, y = neighbors
            if x in non_steiners or y in non_steiners:
                return False

            return self.graph.has_edge(x, y) or self.graph.has_edge(y, x)

        consumptions = 0
        made_progress = True

        while made_progress:
            made_progress = False

            for e in list(self.graph.node_indices()):
                if e == self.super_root_index or e in non_steiners:
                    continue
                if self.reduction_degree(e) < 2:
                    continue

                # Find all degree-2 terminal neighbors of e belonging to same root
                terminal_neighbors = []
                root = None

                for t in list(self.graph.successor_indices(e)):
                    if t not in non_steiners:
                        continue
                    if self.reduction_degree(t) != 2:
                        continue

                    t_root = (
                        t
                        if t in self.terminal_sets
                        else next((r for r, ts in self.terminal_sets.items() if t in ts), None)
                    )
                    if t_root is None:
                        continue

                    if root is None:
                        root = t_root
                    elif t_root != root:
                        root = None
                        break

                    terminal_neighbors.append(t)

                if len(terminal_neighbors) < 2 or root is None:
                    continue

                # Find unambiguous outer endpoints
                outer_endpoints = []
                for t in terminal_neighbors:
                    t_neighbors = self.reduction_neighbors(t)
                    outers = [n for n in t_neighbors if n != e]
                    if len(outers) != 1:
                        # terminal wasn't degree-2
                        continue

                    u = outers[0]
                    if is_unambiguous_outer_endpoint(u, t):
                        outer_endpoints.append(u)

                if not outer_endpoints:
                    continue

                # Certification check: exists at least one u* where w(e) <= w(u*)
                w_e = self.graph[e][PAYLOAD_WEIGHT_KEY]
                for u in outer_endpoints:
                    w_u = self.graph[u][PAYLOAD_WEIGHT_KEY]
                    if w_e <= w_u:
                        # Safe to consume
                        self.consume(e, terminal_neighbors[0])
                        consumptions += 1
                        made_progress = True
                        break

                if made_progress:
                    break

        if consumptions > 0:
            logger.info(f"  consumed {consumptions} enclosed (degree k) same-root Steiners")
            if self.do_debug:
                self.dump_state("reduce_degreek_enclosed_same_root_steiner", consumptions)
                self.validate_reachability()

        return consumptions

    def reduce_enclosed_steiner_clusters(self) -> int:
        """
        Removes Steiner nodes provably absent from every optimal Steiner tree
        realization over every interface subset of recursively decomposed
        Steiner components.

        Large interface components are recursively decomposed across bridge edges
        until the interface count becomes tractable for exact Dreyfus-Wagner
        certification.

        Handles:
        ```
        ∀ ℂ ∈ π₀(G ∖ S)
        I ≔ N(S) ∩ ℂ
        ∀ X ⊆ I, |X| ≥ 2 :
          Wₓ ∈ OPT_ST(X, ℂ)
        k ≔ ℂ ∖ ( ⋃_{X ⊆ I, |X| ≥ 2} Wₓ )
        G ≔ G ∖ k
        ```
        Lemma 1 (Structural Invariance of Enclosed Steiner Cluster Reduction).
        Let S be the set of all non‑Steiner nodes. For each connected component
        ℂ of G ∖ S, let I ≔ N(S) ∩ ℂ be its structural interface.

        For every interface subset X ⊆ I with |X| ≥ 2, let Wₓ ∈ OPT_ST(X, ℂ)
        be an arbitrary optimal Steiner-tree realization inside ℂ. Define:

            k ≔ ℂ ∖ ( ⋃_{X ⊆ I, |X| ≥ 2} Wₓ )

        Then no element of k is required for the existence of at least one
        globally optimal realization of G. Therefore the reduction:

            G ≔ G ∖ k

        preserves:
        • the exact global optimum objective value, and
        • the existence of at least one globally optimal realization.
        """
        logger.trace("reduce_enclosed_steiner_clusters...")

        self.set_edge_weights()

        non_steiners = self.non_steiner_nodes()
        steiners = set(self.graph.node_indices()) - non_steiners
        non_steiners.discard(self.super_root_index)
        if len(steiners) < 3:
            return 0

        g_prime: PyDiGraph = subgraph_stable(self.graph, steiners)
        interfaces = {i for i in steiners if any(n not in steiners for n in self.reduction_neighbors(i))}
        if len(interfaces) < 2:
            return 0

        # self.dump_graph("reduce_enclosed_steiner_clusters", g_prime)

        # This is an node weighted undirected graph...
        g_steiner_undir = nx.Graph()
        for i in g_prime.node_indices():
            g_steiner_undir.add_node(i, weight=g_prime[i][PAYLOAD_WEIGHT_KEY])
        g_steiner_undir.add_edges_from(g_prime.edge_list())  # ignores multi-edges

        removables = set()

        comps = nx.connected_components(g_steiner_undir)
        for comp_nodes in comps:
            comp = set(comp_nodes)
            comp_interfaces = comp & interfaces
            if len(comp_interfaces) < 2:
                continue

            removable = self._reduce_steiner_cluster_dreyfus_wagner(
                component=comp,
                interfaces=comp_interfaces,
                interface_limit=MAX_ENCLOSED_STEINER_INTERFACES,
            )
            removables.update(removable)

        if removables:
            # self.dump_graph("pre-remove enclosed steiner cluster nodes", self.graph)
            self.graph.remove_nodes_from(removables)
            logger.info(f"  removed {len(removables)} Steiner cluster nodes")
            # self.dump_graph("post-remove enclosed steiner cluster nodes", self.graph)
            if self.do_debug:
                logger.trace(f"  {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_enclosed_steiner_clusters_new", len(removables))
                self.validate_reachability()

        return len(removables)

    def _reduce_steiner_cluster_dreyfus_wagner(
        self,
        component: set[int],
        interfaces: set[int],
        interface_limit: int,
    ) -> set[int]:
        """
        Recursive bounded-interface exact certification.

        Components exceeding the interface limit are recursively split.
        Split endpoints become interfaces in their respective child components.
        """
        # logger.trace("  reduce_steiner_cluster_dreyfus_wagner...")

        k = len(interfaces)
        if k <= 1:
            return set()

        # Recursive decomposition phase
        if k > interface_limit:
            split = self._split_steiner_component(component, interfaces)
            if split is None:
                # No valid decomposition found - refuse certification.
                return set()

            left_nodes, right_nodes, bridges_u, bridges_v = split

            left_interfaces = (interfaces | bridges_u | bridges_v) & left_nodes
            right_interfaces = (interfaces | bridges_u | bridges_v) & right_nodes

            removable = set()
            removable |= self._reduce_steiner_cluster_dreyfus_wagner(
                component=left_nodes,
                interfaces=left_interfaces,
                interface_limit=interface_limit,
            )
            removable |= self._reduce_steiner_cluster_dreyfus_wagner(
                component=right_nodes,
                interfaces=right_interfaces,
                interface_limit=interface_limit,
            )
            return removable

        # Preparation phase for Dreyfus-Wagner algorithm
        local_nodes = list(component)
        adj_map: dict[int, set[int]] = {n: set() for n in local_nodes}
        node_weight_map: dict[int, int] = {}

        # Build adjacency map
        for u in local_nodes:
            nbrs = self.reduction_neighbors(u)
            for v in nbrs:
                if v in component:
                    adj_map[u].add(v)
            node_weight_map[u] = int(self.graph[u][PAYLOAD_WEIGHT_KEY])

        def steiner_tree_nodes(terminals: tuple[int, ...]) -> set[int]:
            """Returns the exact optimal Steiner tree topology nodes."""
            terminals_list = list(terminals)
            # Use the first terminal arbitrarily as the root/source node context
            cost, _d_nodes, solution_nodes = nwst_dw.solve_nwst(
                adj_map, node_weight_map, terminals_list, terminals_list[0]
            )
            if cost == 18446744073709551615:  # Matches Rust u64::MAX representation for INF
                return set(terminals)
            return set(solution_nodes)

        # Build KEEP set: nodes that appear in some optimal routing
        keep: set[int] = set()
        interface_list = sorted(interfaces)

        # Steiner trees for all interface subsets up to interface_limit
        for subset_mask in range(1, 1 << len(interface_list)):
            subset = tuple(interface_list[i] for i in range(len(interface_list)) if subset_mask & (1 << i))
            if len(subset) <= interface_limit:
                nodes = steiner_tree_nodes(subset)
                keep.update(nodes)
            else:
                logger.warning(f"    skipping interface subset with {len(subset)} interfaces...")

        # Final component removals: unneeded internal nodes
        return (component - interfaces) - keep

    def _split_steiner_component(
        self,
        component: set[int],
        interfaces: set[int],
    ) -> tuple[set[int], set[int], set[int], set[int]] | None:
        """
        Split a Steiner cluster component into two child components.
        """
        # logger.trace("      split_steiner_component...")

        if len(component) < 6 or len(interfaces) < 2:
            return None

        result = self._find_and_split_bridge_edge_components(component)
        if result is not None:
            # logger.trace("      found single edge bridge split")
            left_component, right_component, bridge_u, bridge_v = result
            return left_component, right_component, {bridge_u}, {bridge_v}

        rxG = subgraph_stable(self.graph, component)
        result = self.find_and_split_2_edge_components(rxG, PAYLOAD_WEIGHT_KEY)
        if result is None:
            return None

        # logger.trace("      found 2-edge bridge split")

        left_gsub, right_gsub, bridges_u, bridges_v = result
        left_component = set(left_gsub.node_indices())
        right_component = set(right_gsub.node_indices())

        return left_component, right_component, set(bridges_u), set(bridges_v)

    def _find_and_split_bridge_edge_components(
        self, component: set[int]
    ) -> tuple[set[int], set[int], int, int] | None:
        """
        Finds a bridge edge suitable for recursive decomposition.

        Returns:
            (
                left_component,
                right_component,
                bridge_u,
                bridge_v,
            )

        where removing edge (u, v) disconnects the component.
        """

        # Build undirected adjacency.
        adj = {n: self.reduction_neighbors(n) & component for n in component}

        # Naive bridge search: Correctness first.
        for u in component:
            for v in adj[u]:
                if u > v:
                    continue

                # Temporarily remove edge.
                adj[u].remove(v)
                adj[v].remove(u)

                # Connectivity test.
                seen = set()
                stack = [u]

                while stack:
                    x = stack.pop()
                    if x in seen:
                        continue

                    seen.add(x)
                    stack.extend(adj[x] - seen)

                # Restore edge.
                adj[u].add(v)
                adj[v].add(u)

                if len(seen) == len(component):
                    continue

                left = seen
                right = component - seen
                if not left or not right:
                    continue

                return (left, right, u, v)

        return None

    def find_and_split_2_edge_components(
        self, digraph: rx.PyDiGraph, weight_attr: str = "weight"
    ) -> tuple[rx.PyDiGraph, rx.PyDiGraph, set[int], set[int]] | None:
        """
        Finds 2-edge cuts using index-aligned local subgraphs, safely maps endpoints
        back to the original DiGraph space using scalar lookups, and returns two valid
        subgraphs for the re-entrant DP states.
        """
        # 1. Isolate structural connectivity with a tracked NodeMap
        g_sub, node_map = digraph.subgraph_with_nodemap(digraph.node_indices())
        undirected_g = g_sub.to_undirected(multigraph=False)

        # 2. Extract node weights using local subgraph keys
        node_weights = {i: g_sub[i].get(weight_attr, 1) for i in undirected_g.node_indices()}
        total_graph_weight = sum(node_weights.values())

        # 3. Construct local spanning tree
        tree_edges_list = rx.graph_dfs_edges(undirected_g)
        tree_edges = set()
        adj_tree = defaultdict(list)
        for u, v in tree_edges_list:
            edge_key = (min(u, v), max(u, v))
            tree_edges.add(edge_key)
            adj_tree[u].append(v)
            adj_tree[v].append(u)

        all_edges = [(min(u, v), max(u, v)) for u, v in undirected_g.edge_list()]
        non_tree_edges = [e for e in all_edges if e not in tree_edges]

        # 4. Generate random 64-bit structural hashes
        edge_hashes = {}
        node_xor_accumulator = defaultdict(int)
        for edge in non_tree_edges:
            h = random.getrandbits(64)
            edge_hashes[edge] = h
            u, v = edge
            node_xor_accumulator[u] ^= h
            node_xor_accumulator[v] ^= h

        # 5. Accumulate values up the spanning tree
        visited = set()
        tree_edge_hashes = {}
        subtree_weights = {}
        node_parent = {}

        def dfs_accumulate(u, p=None):
            visited.add(u)
            node_parent[u] = p
            current_xor = node_xor_accumulator[u]
            current_weight = node_weights[u]

            for v in adj_tree[u]:
                if v != p and v not in visited:
                    child_xor, child_weight = dfs_accumulate(v, u)
                    edge_key = (min(u, v), max(u, v))
                    tree_edge_hashes[edge_key] = child_xor
                    current_xor ^= child_xor
                    current_weight += child_weight

            subtree_weights[u] = current_weight
            return current_xor, current_weight

        for node in undirected_g.node_indices():
            if node not in visited:
                dfs_accumulate(node)

        edge_hashes.update(tree_edge_hashes)

        # 6. Group by structural hash
        hash_to_edges = defaultdict(list)
        for edge, h in edge_hashes.items():
            if h != 0:
                hash_to_edges[h].append(edge)

        valid_cuts = [edge_list for edge_list in hash_to_edges.values() if len(edge_list) == 2]

        best_left_graph = None
        best_right_graph = None
        best_cut_left_endpoints = None
        best_cut_right_endpoints = None
        min_balance_diff = float("inf")

        # 7. Evaluate and extract cuts using translated original IDs
        for e1, e2 in valid_cuts:
            # FIX: e1 and e2 are tuples of (u_sub, v_sub). Extract and map individual integers!
            u1_sub, v1_sub = e1
            u2_sub, v2_sub = e2

            u1_orig = node_map[u1_sub]
            v1_orig = node_map[v1_sub]
            u2_orig = node_map[u2_sub]
            v2_orig = node_map[v2_sub]

            # Clone original input graph for validation testing
            test_graph = digraph.copy()

            # Identify all matching anti-parallel directed arcs using original node coordinates
            edges_to_remove = []
            for src, dst in [(u1_orig, v1_orig), (v1_orig, u1_orig), (u2_orig, v2_orig), (v2_orig, u2_orig)]:
                if test_graph.has_edge(src, dst):
                    edges_to_remove.append((src, dst))
            test_graph.remove_edges_from(edges_to_remove)

            # Verify if a real structural partition occurred
            components = rx.weakly_connected_components(test_graph)
            if len(components) < 2:
                continue  # Rejects false hash collisions or edge shortcuts

            # Sort components by their aggregate node weight to isolate the two largest parts
            components = sorted(
                components, key=lambda c: sum(digraph[n].get(weight_attr, 1) for n in c), reverse=True
            )

            left_nodes_orig = next(iter(components))
            right_nodes_orig = []
            for other_comp in components[1:]:
                right_nodes_orig.extend(list(other_comp))

            # FIX: Construct subgraphs using node indices from the ORIGINAL graph coordinate space
            # This keeps the internal edge wiring and planarity perfectly intact
            left_graph = subgraph_stable(digraph, left_nodes_orig)
            right_graph = subgraph_stable(digraph, right_nodes_orig)

            # Evaluate balance
            left_weight = sum(digraph[n].get(weight_attr, 1) for n in left_nodes_orig)
            right_weight = total_graph_weight - left_weight
            diff = abs(left_weight - right_weight)

            if diff < min_balance_diff:
                min_balance_diff = diff
                best_left_graph = left_graph
                best_right_graph = right_graph
                best_cut_left_endpoints = {u1_orig, u2_orig}
                best_cut_right_endpoints = {v1_orig, v2_orig}

        if (
            best_left_graph is None
            or best_right_graph is None
            or best_cut_left_endpoints is None
            or best_cut_right_endpoints is None
        ):
            return None

        return best_left_graph, best_right_graph, best_cut_left_endpoints, best_cut_right_endpoints

    def reduce_steiner_nodes_by_distance(self) -> int:
        """
        Steiner node reduction by distance.

        For each Steiner node v: if its weight is greater than the longest root -> terminal path, v is dominated and can be removed.
        Also, for each Steiner node v: if the _shortest_ distance to a terminal is greater than the longest root -> terminal path, v is dominated and can be removed.

        Returns number of removed/absorbed nodes.
        """
        logger.trace("reduce_steiner_nodes_by_distance...")

        self.set_edge_weights()
        non_steiner_nodes = self.non_steiner_nodes()
        min_max_drt = self.min_max_rt_distance

        removables = []
        removed_by_weight = 0
        removed_by_distance = 0

        for v in self.graph.node_indices():
            if v in non_steiner_nodes or v == self.super_root_index:
                continue

            if self.graph[v][PAYLOAD_WEIGHT_KEY] > min_max_drt:
                logger.trace(f"  => removing Steiner node {self.get_node_key(v)} by weight")
                removables.append(v)
                removed_by_weight += 1
                continue

            min_drt = float("inf")
            for r, terminals in self.terminal_sets.items():
                for t in terminals | {r}:
                    min_drt = min(min_drt, self.shortest_path_length(v, t))

            if min_drt > min_max_drt:
                logger.trace(f"  => removing Steiner node {self.get_node_key(v)} by distance")
                removables.append(v)
                removed_by_distance += 1
                continue

        if removables:
            self.remove_nodes_from(removables)
            logger.info(
                f"  => removed {len(removables)} Steiner nodes [({min_max_drt}) by weight: {removed_by_weight}, by distance: {removed_by_distance}]"
            )
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(n) for n in removables)}")
                self.dump_state("reduce_steiner_nodes_by_distance", len(removables))
                self.validate_reachability()

        return len(removables)

    def _local_dijkstra(self, source: int, radius: int, weight: dict[int, int]) -> dict[int, int]:
        import heapq

        dist: dict[int, int] = {source: weight[source]}
        heap = [(weight[source], source)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if d > radius:
                break  # stop expanding beyond radius

            for v in self.graph.neighbors(u):
                nd = d + weight[v]
                if nd < dist.get(v, float("inf")) and nd <= radius:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return dist

    def _collect_local_demand_neighborhood(
        self,
        center: int,
        demand_nodes: set[int],
        weight: dict[int, int],
        radius: int,
    ) -> set[int]:
        dist = self._local_dijkstra(center, radius, weight)
        return {u for u in dist if u in demand_nodes}

    def solve_isolated_roots_as_trees(self) -> int:
        """
        Singleton root arbor reduction by interactivity isolation.

        Any isolated root is solved as a single tree. The witness of the tree
        is then consumed from the root outward, leaving the root node in the
        graph.

        NOTE: This function does not handle super terminals as roots.
        Returns number of collapsed nodes.
        """
        logger.trace("solve_isolated_roots_as_trees...")

        self.set_edge_weights()
        interactivity_graph = self._interactivity_graph(self.graph, self.terminal_sets)
        isolates = rx.isolates(interactivity_graph)
        if not isolates:
            return 0
        logger.warning(f"      solving {len(isolates)} isolated trees...")

        all_consumed = set()

        # NOTE: super terminals can not be handled as isolates, but they can be part of an isolates solution...
        sr_index = self.super_root_index

        for root in isolates:
            if root == sr_index:
                continue

            if 1 + len(self.terminal_sets[root]) <= DW_MAX_TREE_TERMINALS:
                witnessed_nodes = self.solve_root_with_dw(root)
            else:
                witnessed_nodes = self.solve_root_as_tree(root)

            # Consume all witnessed nodes, from the root outward; leaving the root node in the graph.
            consumed = set()
            witnessed_to_consume = set(witnessed_nodes)
            while len(witnessed_to_consume) > 0:
                neighbors = self.reduction_neighbors(root)
                neighbors.discard(self.super_root_index)
                neighbors = neighbors & witnessed_to_consume
                if len(neighbors) == 0:
                    break

                for n in neighbors:
                    self.consume(n, root)
                    consumed.add(n)
                    witnessed_to_consume.discard(n)

            all_consumed.update(consumed)

            if self.do_debug:
                logger.trace(f"    witnessed: {sorted(self.get_node_key(n) for n in witnessed_nodes)}")

            # Handle any settled super-root terminals
            if self.super_root_index is not None:
                settled_super_terminals = {
                    i for i in self.terminal_sets[self.super_root_index] if i in witnessed_nodes
                }
                for t in settled_super_terminals:
                    self.terminal_sets[self.super_root_index].remove(t)
                if len(self.terminal_sets[self.super_root_index]) == 0:
                    self.terminal_sets.pop(self.super_root_index)
                    self.super_root_index = None

            # Lastly remove the terminal set cluster
            self.terminal_sets.pop(root)

        if all_consumed:
            logger.debug(f"  => consumed {len(all_consumed)} witnessed nodes")
            if self.do_debug:
                logger.trace(f"    consumed: {sorted(self.get_node_key(n) for n in all_consumed)}")
                self.dump_state("solve_isolated_roots_as_trees", len(all_consumed))
                self.validate_reachability()

        return len(all_consumed)

    def solve_as_tree_super(self):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("    solve_as_tree_super...")

        if self.super_root_index is None:
            return 0

        # Pass thru to MIP (or, once implemented, let the solve as pass deal with this)
        if len(self.terminal_sets) == 1 and len(self.terminal_sets[self.super_root_index]) == 1:
            return 0

        complexity = self.dw_remaining_complexity()

        self.set_edge_weights()

        # --- Collect remaining super terminal candidate roots ---
        # NOTE: If there are any remaining super terminals we must not remove them from the graph
        # _nor_ allow them to become isolated terminals. Therefore, we need to identify the
        # demand sinks (roots, potential roots and terminals) in the neighborhood of each
        # super terminal and add them to the candidate roots for that super terminal.
        candidate_roots: dict[int, set[int]] = defaultdict(set)

        if self.super_root_index in self.terminal_sets:
            roots = set(self.terminal_sets.keys())
            terminal_to_root_map = {v: k for k in roots for v in self.terminal_sets[k]}

            weight_map = {i: self.graph[i][PAYLOAD_WEIGHT_KEY] for i in self.graph.node_indices()}

            for st in self.terminal_sets[self.super_root_index]:
                covered = self._collision_envelope(st, weight_map, self._global_min_max_drt * 2)

                for v in covered:
                    if v in roots:
                        candidate_roots[st].add(v)
                        continue

                    if v in terminal_to_root_map:
                        candidate_roots[st].add(terminal_to_root_map[v])
                        continue

                    # Otherwise it is a potential root...
                    if (
                        v in self.potential_roots
                        and self.shortest_path_length(st, v) <= self._global_min_max_drt
                    ):
                        candidate_roots[st].add(v)

        # --- Solve as tree ---
        start_time = time()
        solution = self.solve_with_scipstp_super(candidate_roots)
        end_time = time()

        removables = set(self.graph.node_indices()) - set(solution)

        # Regardless of the disposition of the super root cluster we don't want to remove it from the graph
        # as this is handled during root reduction.
        if self.super_root_index in removables:
            removables.remove(self.super_root_index)

        removals = len(removables)

        if removals > 0:
            if self.do_debug:
                self.validate_reachability()
                self.dump_graph("before solving as tree")

            duration = end_time - start_time

            num_nodes = self.graph.num_nodes()
            num_roots = len(self.terminal_sets)
            num_terminals = len(set().union(*self.terminal_sets.values()))
            num_super_terminals = len(self.terminal_sets[self.super_root_index])
            logger.warning(
                f"    solved as tree in {duration:.2f}s with complexity {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |st|: {num_super_terminals}, |n|: {num_nodes}"
            )

            self.graph.remove_nodes_from(removables)
            self._solved_as_tree = True

            logger.info(f"    removed {removals} dead tree solution nodes")

            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("solve_as_tree_super", removals)
                self.dump_graph("after solving as tree")
                self.validate_reachability()

        else:
            logger.error("    failed to solve as tree")

        return removals

    def solve_as_tree(self):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_as_tree...")

        if self.super_root_index is not None:
            return 0

        complexity = self.dw_remaining_complexity()
        num_nodes = self.graph.num_nodes()
        num_roots = len(self.terminal_sets)
        num_terminals = len(set().union(*self.terminal_sets.values()))

        # Exludes the MIP 'sweet spot'
        use_scip = self.terminal_sets and (num_nodes > 110 or num_roots < 6)
        start_time = time()
        if use_scip:
            logger.warning("      solving as tree composite")
            G = self.graph.copy()
            if self.super_root_index in self.terminal_sets:
                G.remove_node(self.super_root_index)
            coverage_sets = {r: v for r, v in self.terminal_sets.items() if r != self.super_root_index}

            (
                solution,
                _cost,
                _valid_blocks,
                _block_results,
                _block_costs,
                _dp_block_results,
                _dp_block_costs,
            ) = self.solve_as_tree_composite(G, coverage_sets)

        else:
            logger.warning("      fall-thru: solving as MCF MIP using HiGHS solver...")
            return 0
        end_time = time()

        removables = set(self.graph.node_indices()) - set(solution)
        removals = len(removables)

        if removals > 0:
            duration = end_time - start_time
            logger.warning(
                f"    solved as tree in {duration:.2f}s with complexity {complexity}, |r|: {num_roots}, |t|: {num_terminals}, |n|: {num_nodes}"
            )

            if self.do_debug:
                print("pre-tree-solution graph dead node removal...")
                self.validate_reachability()
                self.dump_graph("before solving as tree")

            self.graph.remove_nodes_from(removables)

            logger.info(f"    removed {removals} dead tree solution nodes")
            if self.do_debug:
                logger.trace(f"    {sorted(self.get_node_key(i) for i in removables)}")
                self.dump_state("solve_as_tree", removals)
                self.dump_graph("after solving as tree")
                self.validate_reachability()

        else:
            logger.error("    failed to solve as tree")

        return removals

    # MARK: Tree Solvers

    def solve_root_with_dw(self, root: int):
        """Solves a single root from the remaining reduced state graph."""
        import nwst_dw

        logger.trace("solve_root_as_dw...")

        adj_map = {
            u: set(self.graph.neighbors(u)) for u in self.graph.node_indices() if u != self.super_root_index
        }
        node_weight_map = self.node_weight_map
        if self.super_root_index is not None:
            node_weight_map.pop(self.super_root_index)

        solver = nwst_dw.Solver(adj_map, node_weight_map)
        terminals_list = list(self.terminal_sets[root] | {root})

        self.solved_trees += 1
        cost, _, solution_nodes = solver.solve(terminals_list, root)
        if cost == 18446744073709551615:
            raise RuntimeError("Unsolvable Steiner tree!")

        if solution_nodes:
            logger.debug(
                f"    solved tree for root {self.get_node_key(root)} containing {len(terminals_list)} terminals with cost {cost}"
            )
            logger.trace(f"    solution: {sorted(self.get_node_key(n) for n in solution_nodes)}")

        return solution_nodes

    def solve_root_as_tree(self, root: int) -> set[int]:
        """Solves a single isolated root from the remaining reduced state graph.

        NOTE: An isolated root (per the interactivity-graph isolation check) has
        no possible interaction with any other root, so there is no partition
        or decomposition question to answer here -- unlike solve_as_tree_composite,
        this is always exactly one block: the root's own flattened terminal set,
        solved as a single tree.
        """
        logger.trace("solve_root_as_tree...")

        terminals_list = sorted(self.terminal_sets[root] | {root})
        logger.trace(f"    solving isolated root {root} with {len(terminals_list)} terminals...")

        self.solved_trees += 1

        root_component_index_map, component_data = _connected_component_mappings(
            self.graph, set(self.terminal_sets.keys())
        )
        comp_map = component_data[root_component_index_map[root]]

        # NOTE: solve_tree auto switches between DW and scipstp based on tree complexity,
        #       matching solve_as_tree_composite's dispatch behavior.
        _, cost, mask = solve_tree(
            TreeProblem(
                instance_id=self.instance_id,
                block_key=(root,),
                terminals=terminals_list,
                adj_map=comp_map["adj_map"],
                node_weight_map=self.node_weight_map,
                node_index_map=self.node_index_map,
                dimacs_id_map=comp_map["dimacs_id_map"],
                inv_dimacs_id_map=comp_map["inv_dimacs_id_map"],
                enable_super_root_index=0,
                do_debug=self.do_debug,
                mip_validation=False,
            )
        )

        logger.trace(f"    solved isolated root {root} with cost {cost}")
        best_solution = _unmask_solution(self.graph, mask)

        if best_solution:
            logger.debug(
                f"    solved for terminals: {self.get_node_key(root)}: "
                f"{ {self.get_node_key(n) for n in self.terminal_sets[root]} }"
            )
            logger.debug(
                f"    found best solution (cost: {cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    def _collision_envelope(self, source: int, weight_map: dict[int, int], radius_cap: float) -> set[int]:
        """Dijkstra from source, expanding only through Steiner nodes and freezing each
        frontier branch on its first collision with an existing arbor (any node already
        committed to some root's demand). Mirrors is_potential_violation's stop-at-non-
        Steiner rule, but accumulates the touched envelope and collision points instead
        of returning a bool.

        Returns (envelope, collisions):
        envelope   -- Steiner nodes actually traversed to reach a collision
        collisions -- the first non-Steiner node hit along each frontier branch
        """
        non_steiner = self.non_steiner_nodes()
        dist: dict[int, float] = {source: weight_map[source]}
        heap = [(weight_map[source], source)]
        envelope: set[int] = set()
        collisions: set[int] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if d > radius_cap:
                break
            if u != source and u in non_steiner:
                collisions.add(u)
                continue  # frozen: don't expand past a collision

            envelope.add(u)
            for v in self.reduction_neighbors(u):
                nd = d + weight_map[v]
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return collisions

    def _filter_uncertified_st_edges(
        self,
        interactivity_edges: list[tuple[int, int, float]],
        candidate_roots: dict[int, set[int]],
        surviving_sts: set[int],
    ) -> list[tuple[int, int, float]]:
        """Drops st<->root edges the gap/drt test reports but the collision-envelope search
        already disproved. gap1+gap2 <= drt1+drt2 is a sufficient-condition heuristic and can
        false-positive; candidate_roots[st] is an exact containment proof (first-collision
        boundary), so non-membership there certifies non-interactivity regardless of what
        the gap test claims.
        """
        filtered = []
        for u, v, gap in interactivity_edges:
            u_is_st, v_is_st = u in surviving_sts, v in surviving_sts
            if u_is_st and not v_is_st and v not in candidate_roots[u]:
                continue
            if v_is_st and not u_is_st and u not in candidate_roots[v]:
                continue
            filtered.append((u, v, gap))
        return filtered

    def solve_with_scipstp_super(
        self,
        candidate_roots: dict[int, set[int]],
    ):
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.warning("    solve_with_scipstp_super...")

        assert self.super_root_index is not None, "super root index must be set"

        # --- Pre-solve non-super-terminal-blocks ---
        # NOTE: any block in valid blocks but not in block results was dominated prior to solving.
        non_super_G = self.graph.copy()
        non_super_G.remove_node(self.super_root_index)
        non_super_root_coverage_representatives = {
            k: v for k, v in self.terminal_sets.items() if k != self.super_root_index
        }
        (
            _,
            _,
            non_super_valid_blocks,
            _non_super_block_results,
            _non_super_block_costs,
            non_super_dp_block_results,
            non_super_dp_block_costs,
        ) = self.solve_as_tree_composite(non_super_G, non_super_root_coverage_representatives)
        non_super_valid_blocks = set().union(*non_super_valid_blocks.values())
        non_super_dominated_blocks = set(non_super_valid_blocks) - set(non_super_dp_block_results.keys())
        # print(
        #     f"    presolve results: num_valid: {len(non_super_valid_blocks)}, num_dominated: {len(non_super_dominated_blocks)}, num_results: {len(non_super_dp_block_results)}"
        # )

        # Even though scipstp NWSTP is fully a node weighted setup the rustworkx graph isn't pickleable
        # so we need to pass in the adjacency map and node weights
        adj_map = {u: sorted(self.reduction_neighbors(u)) for u in self.graph.node_indices()}
        node_weight_map = {u: self.graph[u][PAYLOAD_WEIGHT_KEY] for u in self.graph.node_indices()}

        roots = list(set(self.terminal_sets.keys()) - {self.super_root_index})
        super_terminals = sorted(candidate_roots.keys())
        coverage_entities = roots + super_terminals

        # Root reachability for component generation...
        logger.trace("      building reachability matrix...")
        reachable = {r: {r} for r in coverage_entities}
        for i, u in enumerate(coverage_entities):
            for covered_entity in coverage_entities[i + 1 :]:
                if rx.has_path(self.graph, u, covered_entity):
                    reachable[u].add(covered_entity)
                    reachable[covered_entity].add(u)

        # Connected root components generation...
        components = []
        remaining_roots = set(coverage_entities)
        while remaining_roots:
            root = next(iter(remaining_roots))
            component = sorted(reachable[root])
            components.append(component)
            remaining_roots -= reachable[root]

        logger.trace(
            f"      identified {len(components)} reachability component(s): {[len(c) for c in components]}"
        )

        # For index masking...
        nodes_list = list(self.graph.node_indices())
        node_index = {u: i for i, u in enumerate(nodes_list)}

        # DIMACS node ids are 1-based, contiguous
        dimacs_id = {u: i + 1 for i, u in enumerate(nodes_list)}
        inv_dimacs_id = {i + 1: u for i, u in enumerate(nodes_list)}

        # Partition filtering...
        logger.trace("    generating valid partition blocks...")

        # Since the partition DP reconstructs the optimal partitioning, we only need
        # to generate the valid blocks. Any valid block must be wholly contained
        # within a single reachability component.
        valid_blocks = set()
        for component in components:
            logger.trace(f"      generating blocks for component size={len(component)}...")
            k = len(component)
            for mask in range(1, 1 << k):
                block = tuple(component[i] for i in range(k) if mask & (1 << i))
                valid_blocks.add(block)
        logger.trace(f"    generated valid blocks ({len(valid_blocks)})")

        # Filter valid blocks by removing all non-super dominated blocks...
        valid_blocks -= non_super_dominated_blocks
        logger.trace(
            f"    filtered {len(non_super_dominated_blocks)} valid blocks by non-super blocks... ({len(valid_blocks)}) remaining"
        )

        skipped_presolved_block_count = 0
        super_terminals_set = set(super_terminals)

        # --- Generate tasks ---
        singleton_tasks = []
        composite_tasks = []

        # Sort all remaining valid blocks by block key length and block key
        valid_blocks = sorted(valid_blocks, key=lambda b: (len(b), b))

        for block_key in valid_blocks:
            terminals = set()

            # A super terminal behaves as a terminal in the presence of another root,
            # yet it behaves as a root with a single terminal (the super root) when
            # there are no other roots. This enforces potential root transit when the
            # super terminals are considered in isolation.
            sr_index = 0  # disabled
            if all(i in super_terminals for i in block_key):
                terminals.add(self.super_root_index)
                sr_index = self.super_root_index

            for k in block_key:
                terminals.add(k)
                if k not in super_terminals:
                    terminals.update(self.terminal_sets[k])

            if not terminals & super_terminals_set:
                skipped_presolved_block_count += 1
                continue

            terminals = sorted(terminals)
            task = (block_key, terminals, sr_index)
            if len(block_key) == 1 or len(block_key) == 2 and sr_index == block_key[1]:
                singleton_tasks.append(task)
            else:
                composite_tasks.append(task)

        logger.warning(f"    skipped resolving presolved blocks: {skipped_presolved_block_count}")
        num_singleton_tasks = len(singleton_tasks)
        num_composite_tasks = len(composite_tasks)
        print(
            f"    num tasks: {num_singleton_tasks + num_composite_tasks} (singletons: {num_singleton_tasks}) (composites: {num_composite_tasks})"
        )

        # --- Solve singleton tasks ---
        problem_generator = (
            TreeProblem(
                instance_id=self.instance_id,
                block_key=task[0],
                terminals=task[1],
                adj_map=adj_map,
                node_weight_map=node_weight_map,
                node_index_map=node_index,
                dimacs_id_map=dimacs_id,
                inv_dimacs_id_map=inv_dimacs_id,
                enable_super_root_index=task[2],
                do_debug=self.do_debug,
                mip_validation=False,
            )
            for task in singleton_tasks
        )
        num_problems = len(singleton_tasks)

        block_results = dict(non_super_dp_block_results)
        block_costs = dict(non_super_dp_block_costs)
        self._solve_treeproblem_tasks(
            problem_generator, num_problems, results_dest=block_results, costs_dest=block_costs
        )

        # --- Composite representatives interactivity setup ---
        # Super terminal envelopes... super root
        st_interactivity_coverage_sets: CoverageSets = defaultdict(set)
        for t in super_terminals:
            paths = rx.all_shortest_paths(
                self.graph, t, self.super_root_index, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY]
            )
            trimmed = [p[:-1] for p in paths]  # drop the synthetic sr hop
            logger.trace(f"st: {t} -> {sr_index}: {[[self.get_node_key(v) for v in p] for p in trimmed]}")
            st_interactivity_coverage_sets[t] = {p[-1] for p in trimmed}

        # Super terminal envelopes... candidate roots
        for t in super_terminals:
            st_interactivity_coverage_sets[t] |= set(candidate_roots[t])

        interactivity_composite_coverage_sets = dict(non_super_root_coverage_representatives)

        for k, covered_entities in st_interactivity_coverage_sets.items():
            interactivity_composite_coverage_sets[k] = covered_entities

        # Special case handling for path unions...
        G = self.graph

        # Drop all instances of super root from all coverage sets
        for s in interactivity_composite_coverage_sets.values():
            s.discard(self.super_root_index)

        coverage_representatives = set(interactivity_composite_coverage_sets.keys())
        node_weight_map = {u: G[u][PAYLOAD_WEIGHT_KEY] for u in G.node_indices()}

        # Union sets of all nodes for all shortest paths for each t -> r for each t in terminal_set r
        arbor_rt_all_shortest_path_unions = {r: set() for r in interactivity_composite_coverage_sets}
        for r, terminals in interactivity_composite_coverage_sets.items():
            if r not in coverage_representatives:
                continue
            for t in terminals:
                paths = rx.all_shortest_paths(G, t, r, weight_fn=lambda e: e[PAYLOAD_WEIGHT_KEY])
                arbor_rt_all_shortest_path_unions[r].update(*paths)

        interactivity_edges = get_interactivity_edges(
            non_super_G,
            interactivity_composite_coverage_sets,
            node_weight_map,
            arbor_rt_all_shortest_path_unions,
        )
        filtered_interactivity_edges = self._filter_uncertified_st_edges(
            interactivity_edges, candidate_roots, set(super_terminals)
        )
        logger.warning(
            f"    filtered ({len(interactivity_edges)} -> {len(filtered_interactivity_edges)}) interactivity edges..."
        )
        interactivity_edges = filtered_interactivity_edges

        interactivity_graph = subgraph_stable(G, coverage_representatives)
        interactivity_graph.remove_edges_from(interactivity_graph.edge_list())
        interactivity_graph.add_edges_from(interactivity_edges)
        composite_interactivity_graph = interactivity_graph

        if self.do_debug:
            print(f"      max_drt = {self.min_max_rt_distance}")
            tmp_map = {i: (n, self.get_node_key(n)) for i, n in enumerate(interactivity_graph.node_indices())}
            print(f"      interactivity_graph.num_nodes = {interactivity_graph.num_nodes()}")
            print(f"      interactivity_graph.num_edges = {interactivity_graph.num_edges()}")
            print(
                f"      num weakly connected components = {rx.number_weakly_connected_components(interactivity_graph)}"
            )
            print(
                f"      num strongly connected components = {rx.number_strongly_connected_components(interactivity_graph)}"
            )
            print("      interactivity_graph.map:")
            for k, v in tmp_map.items():
                print(f"        {k}: {v}")

            adj_matrix = rx.adjacency_matrix(interactivity_graph)

            interactive_roots: dict[int, list[int]] = {}
            for root_i, row in enumerate(adj_matrix):
                _root = tmp_map[root_i][0]
                root_wp = tmp_map[root_i][1]
                interactive_roots[root_wp] = []
                for col_i, weight in enumerate(row):
                    if weight:
                        col_wp = tmp_map[col_i][1]
                        interactive_roots[root_wp].append(col_wp)

            print("      interactive_roots:")
            for root, cols in interactive_roots.items():
                print(f"        {root}: {cols}")

        # --- Filter dominated composite tasks by connectivity ---
        num_composite_tasks = len(composite_tasks)
        surviving_composite_tasks = []
        for task in composite_tasks:
            block_key = task[0]
            subIG = composite_interactivity_graph.subgraph(block_key)
            if not rx.is_strongly_connected(subIG):
                continue
            surviving_composite_tasks.append(task)

        logger.warning(
            f"    Connectivity filtering: num surviving composite tasks: {len(surviving_composite_tasks)} of ({num_composite_tasks})"
        )
        composite_tasks = surviving_composite_tasks

        # --- Filter dominated composite tasks by max_dist/MST ---
        # Phase 2 (Wave 2 gate) - filter by dominance using max_dist/MST bound and primitive cost
        # NOTE: OPT >= W_MST / (2 - 2/k) (KMB). If that lower bound already exceeds the cheapest
        #       decomposition we can prove right now (singleton sum), no joint solve can beat it.
        #       Strict '>' only -- ties fall through to the solver.
        #       (Same as the post-solve dominance check, since a tying composite may still share
        #       more structure with the rest of the forest than any decomposition would.)
        # --- Filter dominated composite tasks by max_dist/MST ---
        terminal_to_terminal_distances = {}
        all_terminals = sorted(
            set(non_super_root_coverage_representatives.keys()).union(
                *non_super_root_coverage_representatives.values()
            )
            | set(super_terminals)
        )
        for i, ti in enumerate(all_terminals):
            for tj in all_terminals[i + 1 :]:
                d = self.shortest_path_length(ti, tj)
                terminal_to_terminal_distances[(ti, tj)] = d
                terminal_to_terminal_distances[(tj, ti)] = d

        composite_tasks = _retain_dominant_composite_tasks_by_distance(
            composite_tasks, block_costs, terminal_to_terminal_distances, self.super_root_index
        )

        # --- Solve composite tasks ---
        logger.warning(f"    Solving {len(composite_tasks)} composite tasks")
        problem_generator = (
            TreeProblem(
                instance_id=self.instance_id,
                block_key=task[0],
                terminals=task[1],
                adj_map=adj_map,
                node_weight_map=node_weight_map,
                node_index_map=node_index,
                dimacs_id_map=dimacs_id,
                inv_dimacs_id_map=inv_dimacs_id,
                enable_super_root_index=task[2],
                do_debug=self.do_debug,
                mip_validation=False,
            )
            for task in composite_tasks
        )
        num_problems = len(composite_tasks)
        self.solved_trees += num_problems

        self._solve_treeproblem_tasks(
            problem_generator, num_problems, results_dest=block_results, costs_dest=block_costs
        )

        best_solution, best_cost, _block_results, _block_costs = _solve_composite_blocks_dp(
            self.graph, set(coverage_entities), block_results, block_costs
        )

        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.debug(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return best_solution

    def solve_as_tree_composite(
        self, G: PyDiGraph, coverage_sets: dict[int, set[int]]
    ) -> tuple[
        set[int],
        int | float,
        dict[int, set[tuple[int, ...]]],
        dict[tuple[int, ...], int],
        dict[tuple[int, ...], int],
        dict[tuple[int, ...], int],
        dict[tuple[int, ...], int],
    ]:
        """Solves the remaining stable reduced state graph by taking the best solution from the composite instances."""
        logger.trace("solve_with_scipstp...")

        self.dump_graph("solve_with_scipstp: pre-solve")

        # Sometimes all that is left are super terminals and there's nothing to be done here...
        if not coverage_sets:
            return set(), float("inf"), {}, {}, {}, {}, {}

        coverage_representatives = set(coverage_sets.keys())

        _root_component_index_map, component_data = _connected_component_mappings(G, coverage_representatives)
        interactivity_graph = self._interactivity_graph(G, coverage_sets)
        weight_map = {i: G[i][PAYLOAD_WEIGHT_KEY] for i in G.node_indices()}

        # Phase 1 - block preparation
        valid_blocks, num_blocks, num_candidate_blocks = _generate_valid_partition_blocks(
            component_data, interactivity_graph
        )
        block_terminals = _map_block_terminals(coverage_sets, valid_blocks)

        # --- Wave 1: singletons. These are the atoms of every decomposition --
        # there's nothing to prune them against, so they always get solved.
        singleton_tasks = [
            (cc_i, block_key, block_terminals[(cc_i, block_key)])
            for cc_i, blocks in valid_blocks.items()
            for block_key in blocks
            if len(block_key) == 1
        ]

        problem_generator = _problem_generator(
            self.instance_id,
            component_data,
            weight_map,
            singleton_tasks,
            do_debug=self.do_debug,
            mip_validation=False,
        )

        # NOTE: Long lived collections throughout the solving phases
        block_results = {}
        block_costs = {}

        # Solve all singletons for usage in composite block filtering (Wave 2)
        logger.warning(
            f"    sequentially solving {len(singleton_tasks)} singleton blocks of {num_candidate_blocks} ({num_blocks / num_candidate_blocks:.2%})..."
        )

        self.solved_trees += len(singleton_tasks)
        for block_n, problem in enumerate(problem_generator, start=1):
            logger.trace(f"    solving ({block_n}/{num_blocks}) block {problem.block_key}...")

            # NOTE: solve_tree auto switches between DW and scipstp based on tree complexity
            block_key, cost, mask = solve_tree(problem)

            logger.trace(
                f"    solved ({block_n}/{num_blocks}) block {block_key} containing {len(problem.terminals)} terminals with cost {cost}"
            )

            block_results[block_key] = mask
            block_costs[block_key] = cost

        # Phase 2 (Wave 2 gate) - filter by dominance using max_dist/MST bound and primitive cost
        # NOTE: OPT >= W_MST / (2 - 2/k) (KMB). If that lower bound already exceeds the cheapest
        #       decomposition we can prove right now (singleton sum), no joint solve can beat it.
        #       Strict '>' only -- ties fall through to the solver.
        #       (Same as the post-solve dominance check, since a tying composite may still share
        #       more structure with the rest of the forest than any decomposition would.)
        terminal_to_terminal_distances = {}
        all_terminals = sorted(set(coverage_sets.keys()).union(*coverage_sets.values()))
        for i, ti in enumerate(all_terminals):
            for tj in all_terminals[i + 1 :]:
                d = self.shortest_path_length(ti, tj)
                terminal_to_terminal_distances[(ti, tj)] = d
                terminal_to_terminal_distances[(tj, ti)] = d

        # block_costs serves as the primitive block costs...
        surviving_tasks = _retain_dominant_blocks_by_distance(
            valid_blocks, block_terminals, block_costs, terminal_to_terminal_distances
        )

        # Wave 2: Surviving valid block solving...
        self.solved_trees += len(surviving_tasks)
        problem_generator = _problem_generator(
            self.instance_id,
            component_data,
            weight_map,
            surviving_tasks,
            do_debug=self.do_debug,
            mip_validation=False,
        )
        num_problems = len(surviving_tasks)

        logger.warning(
            f"    solving {num_problems} surviving of {num_blocks} unique valid blocks of {num_candidate_blocks} ({num_blocks / num_candidate_blocks:.2%})..."
        )

        self._solve_treeproblem_tasks(
            problem_generator, num_problems, results_dest=block_results, costs_dest=block_costs
        )

        best_solution, best_cost, dp_block_results, dp_block_costs = _solve_composite_blocks_dp(
            G, coverage_representatives, block_results, block_costs
        )
        if best_solution:
            logger.debug(
                f"    solved for terminals: { {self.get_node_key(r): {self.get_node_key(n) for n in ts} for r, ts in self.terminal_sets.items()} }"
            )
            logger.trace(
                f"    found best solution (cost: {best_cost}): {sorted(self.get_node_key(n) for n in best_solution)}"
            )

        return (
            best_solution,
            best_cost,
            valid_blocks,
            block_results,
            block_costs,
            dp_block_results,
            dp_block_costs,
        )

    def _solve_treeproblem_tasks(
        self,
        problem_generator: Generator[TreeProblem],
        num_problems: int,
        results_dest: dict[tuple[int, ...], int],
        costs_dest: dict[tuple[int, ...], int],
    ):
        orig_num_results = len(results_dest)

        # Yes, really, empirically it is 42, lol.
        if num_problems >= 42:
            logger.warning("    _solve_treeproblem_tasks: solving concurrently...")

            try:
                with ProcessPoolExecutor(max_workers=14) as executor:
                    chunksize = max(1, min(256, num_problems // 16))
                    logger.warning(f"    chunksize: {chunksize}")

                    # NOTE: solve_tree auto switches between DW and scipstp based on tree complexity
                    results = executor.map(solve_tree, problem_generator, chunksize=chunksize)

                    for completed_count, (block_key, cost, mask) in enumerate(results, start=1):
                        if completed_count % 1_000 == 0:
                            logger.warning(
                                f"    solved ({completed_count}/{num_problems}) {completed_count / num_problems:.2%}..."
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
                    f"    solved ({block_n}/{num_problems}) block {block_key} containing {len(problem.terminals)} terminals with cost {cost}"
                )

                results_dest[block_key] = mask
                costs_dest[block_key] = cost

        logger.warning(
            f"    solved {len(results_dest) - orig_num_results} unique valid blocks .."
        )  # of {num_candidate_blocks} ({len(results_dest) / num_candidate_blocks:.2%})"

    def _interactivity_graph(
        self,
        G: PyDiGraph,
        coverage_sets: dict[int, set[int]],
    ) -> PyDiGraph:
        """Creates the interactivity graph for the given reduced state graph over the given coverage sets."""
        interactivity_graph = _interactivity_graph(G, coverage_sets)

        if self.do_debug:
            tmp_map = {i: (n, self.get_node_key(n)) for i, n in enumerate(interactivity_graph.node_indices())}
            print(f"      interactivity_graph.num_nodes = {interactivity_graph.num_nodes()}")
            print(f"      interactivity_graph.num_edges = {interactivity_graph.num_edges()}")
            print("      interactivity_graph.map:")
            for k, v in tmp_map.items():
                print(f"        {k}: {v}")

            adj_matrix = rx.adjacency_matrix(interactivity_graph)

            interactive_roots: dict[int, list[int]] = {}
            for root_i, row in enumerate(adj_matrix):
                _root = tmp_map[root_i][0]
                root_wp = tmp_map[root_i][1]
                interactive_roots[root_wp] = []
                for col_i, weight in enumerate(row):
                    if weight:
                        col_wp = tmp_map[col_i][1]
                        interactive_roots[root_wp].append(col_wp)

            print("      interactive_roots:")
            for root, cols in interactive_roots.items():
                print(f"        {root}: {cols}")

        return interactivity_graph

    # MARK: Main Loop

    def run_pipeline(self) -> tuple[set[int], dict[int, int]]:
        """Runs the main reduction pipeline."""
        logger.trace("run_pipeline...")

        num_edges_start = self.graph.num_edges()
        num_nodes_start = self.graph.num_nodes()
        num_terminal_roots_start = len(self.terminal_sets)
        num_terminals_start = sum(len(ts) for ts in self.terminal_sets.values()) + num_terminal_roots_start

        # In the original problem space the min max drt represents a global bound that
        # superceeds the min max drt of any terminal set in the reduced problem space as
        # trees collide. If clusters a, b and c initially have a min max drt and two of them
        # are merged to form d, then the min max drt of d would be calculated at a greater
        # value than the global yet d could not actually detour further than the global bound
        # because it is constrained by the beneficial detour of the other two clusters.
        self.set_edge_weights()
        min_max_drt = self.min_max_rt_distance
        logger.info(f"  Global maximum RT distance: {min_max_drt}")

        tmp_num_nodes = self.graph.num_nodes() + 1
        iteration = 0

        while (num_nodes := self.graph.num_nodes()) != tmp_num_nodes:
            # Update self.do_debug flag when log level is externally modified
            active_severity = logger._core.min_level  # ty:ignore[unresolved-attribute]
            self.do_debug = active_severity <= 10

            tmp_num_nodes = num_nodes
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            # MARK: Simple reductions

            # Isolates in the graph are always safe to directly remove
            if isolates := rx.isolates(self.graph):
                self.graph.remove_nodes_from(isolates)
                logger.info(f"  removed {len(isolates)} isolates...")
                logger.trace(f"    {sorted(self.get_node_key(n) for n in isolates)}")

            # Reduce potential roots - this is not a graph mutation it is a potential roots set consolidation only
            self.reduce_potential_roots()

            # Reduce demand roots - this is not a graph mutation it is a terminal set consolidation only
            self.reduce_roots_via_articulation_points()

            # Reduce demand roots - this is not a graph mutation it is a terminal set consolidation only
            self.reduce_demand_roots()

            # Reduce adjacent terminals - this is a terminal reduction by `consume` with terminal set consolidation
            self.reduce_adjacent_terminals()

            # Reduce degree1 steiner nodes - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            self.reduce_degree1_steiner_nodes()

            # Reduce degree1 terminals - this is a Steiner node graph reduction by `remove` only
            self.reduce_degree1_terminals()

            # Reduce degree1 roots - this is a Steiner node graph reduction by `consume` with terminal set consolidation
            self.reduce_degree1_roots()

            if num_nodes != self.graph.num_nodes():
                logger.info("  repeating simple reductions...")
                continue

            if not self.terminal_sets:
                logger.success("  no terminal sets left. All demands are satisfied!")
                break

            # Solve isolated roots - this is a terminal set and Steiner node reduction by `consumption` only
            if self.solve_isolated_roots_as_trees():
                continue

            # MARK: Basic reductions

            # Reduce 2-degree articulation points - this is a Steiner node graph reduction with mixed handling
            self.reduce_degree2_articulation()

            # Reduce 2-degree steiner chains - this is a Steiner node graph reduction by `absorb` only
            self.merge_adjacent_degree2_steiner_chains()

            # Reduce steiner triangle 2-degree legs - this is a Steiner node graph reduction by `absorb` only
            self.reduce_steiner_triangle_degree2_legs()

            # Reduce 2-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            self.reduce_degree2_steiner_dominance()

            # Reduce k-degree steiner dominance - this is a Steiner node graph reduction by `remove` only
            self.reduce_degreek_steiner_dominance()

            # Reduce steiner bridges - this is a Steiner node graph reduction by `consume` only
            self.reduce_steiner_bridges()

            # Reduce Steiner nodes by distance - this is a Steiner node graph reduction by `remove` only
            self.reduce_steiner_nodes_by_distance()

            # Reduce degreek enclosed steiner - this is a Steiner node graph reduction by `consume` only
            self.reduce_degreek_enclosed_steiner()

            # Reduce blocked roots - this is a terminal set consolidation only
            if num_nodes != self.graph.num_nodes() and self.reduce_blocked_roots() > 0:
                # Force a reduction pass if blocked roots were merged.
                tmp_num_nodes = 0
                continue

            # Reduce enclosed steiner clusters - this is a Steiner node graph reduction by `remove` only
            if num_nodes == self.graph.num_nodes():
                self.reduce_enclosed_steiner_clusters()

            # NOTE: The following two reductions are virtually identical to the normal reductions but
            # have super root/terminal support. The are distinct from the normal reductions because
            # they have additional overhead _and_ alter the cascading reductions when used in place of
            # the normal reductions.

            if self.super_root_index is not None and num_nodes != self.graph.num_nodes():
                # reduce_degree2_articulation_super - this is a Steiner node graph reduction by `remove` only
                removed = self.reduce_degree2_articulation_super()

                # reduce_steiner_bridges_super - this is a Steiner node graph reduction by `remove` only
                removed += self.reduce_steiner_bridges_super()

                if removed > 0:
                    tmp_num_nodes = 0
                    continue

            # Solve as tree for super root
            if (
                not self._solved_as_tree
                and num_nodes == self.graph.num_nodes()
                and self.super_root_index is not None
                and self.terminal_sets
            ):
                self.dump_reduction_results(
                    "pre: solve_as_tree_super",
                    num_edges_start,
                    num_nodes_start,
                    num_terminal_roots_start,
                    num_terminals_start,
                )
                if self.solve_as_tree_super() > 0:
                    tmp_num_nodes = 0
                    self.solved_as_tree = True
                    continue

            # Solve as tree for non super root
            if (
                not self._solved_as_tree
                and num_nodes == self.graph.num_nodes()
                and self.super_root_index is None
                and self.terminal_sets
            ):
                self.dump_reduction_results(
                    "pre: solve_as_tree",
                    num_edges_start,
                    num_nodes_start,
                    num_terminal_roots_start,
                    num_terminals_start,
                )
                self.solve_as_tree()
                self.solved_as_tree = True

        # Rebuild pairs
        fixed_nodes_wp = {self.get_node_key(n) for n in self.fixed_nodes}
        reduced_root_pairs_wp: dict[int, int] = {}
        for r, comp in self.terminal_sets.items():
            for t in comp:
                reduced_root_pairs_wp[self.get_node_key(t)] = self.get_node_key(r)

        if self.do_debug:
            self.dump_final_report(
                reduced_root_pairs_wp,
                fixed_nodes_wp,
            )

        self.dump_reduction_results(
            "final",
            num_edges_start,
            num_nodes_start,
            num_terminal_roots_start,
            num_terminals_start,
        )

        if reduced_root_pairs_wp:
            logger.warning(f"  Solving remaining {len(reduced_root_pairs_wp)} pairs via MIP...")

        return fixed_nodes_wp, reduced_root_pairs_wp
