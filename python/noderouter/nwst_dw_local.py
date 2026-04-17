# nwst_dw.py

from collections import defaultdict
from itertools import combinations
import heapq
import math
import time


def DW_dijkstra(graph, vertex_weights, start, vertices):
    """
    Calculates length shortest path between all u, v ∈ V using Dijkstra's algorithm.

    Defines duu = w(u), a path's length is the sum of weights of all of its edges and vertices
    (including endpoint vertices)

    Args:
        graph: Dict[u][v] with edge weights
        vertex_weights: Dictionary mapping vertices to their weights (costs).
        start: The source vertex (str or int).
        vertices: list of vertex ids

    Returns:
        distances: Dict[u][v] mapping to path cost
        predecessors: Dict[u][v] mapping to predecessor of v on the shortest path to u
    """
    # Compute length shortest path between all u, v ∈ V, define duu = w(u)
    # A path’s length is the sum of weights of all of its edges and vertices
    # (including endpoint vertices)
    distances = {v: math.inf for v in vertices}
    predecessors = {v: None for v in vertices}
    distances[start] = vertex_weights.get(start, 0)
    pq = [(distances[start], start)]
    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances[u]:
            continue
        for v, edge_weight in graph.get(u, {}).items():
            new_dist = dist + edge_weight + vertex_weights.get(v, 0)
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))

    return distances, predecessors


def DW_nwst(d_w, vertex_weights, terminals, source):
    r"""
    Calculates the Node-Weighted Steiner Tree connecting source to terminals using the
    Node Weighted Dreyfus Wagner algorithm 1 from
    `Algorithms for node-weighted Steiner tree and maximum-weight connected subgraph` by
    Buchanan et al.

    Args:
        d_w: Path distance weights from u to v (includes edge and node weights, including endpoints)
        vertex_weights: Dictionary mapping vertices to their weights (costs).
        terminals: Set of terminal vertices.
        source: The source vertex (str or int).

    Returns:
        cost: Total cost of the Steiner Tree.
        D: The set of terminals minus source.
        split_info: Dictionary mapping (v, D) to (u, A, D \ A) for leaf nodes or (v, A, D \ A) for interior nodes.
    """

    def w(v):
        return vertex_weights.get(v, 0)

    vertices = set(graph.keys())

    q = {}
    for u in vertices:
        for v in vertices:
            q[(u, tuple([v]))] = d_w[u].get(v, math.inf)

    q_split_info = {}
    for v in vertices:
        for u in terminals:
            q_split_info[(v, tuple([u]))] = q[(v, tuple([u]))]

    terminals_minus_source = terminals - {source}

    k = len(terminals)
    p = {}
    p_split_info = {}  # (A, D_minus_A) for p[(v, D)]
    for i in range(2, k):  # range is [2, k) or |T| - 1
        for D in combinations(terminals_minus_source, i):
            # Cost p(v, D) of connecting v to D in a solution in which v is an interior (non-leaf) node
            # if v is indeed an interior node in a minimum cost NWST that connects D ∪ {v},
            # then the algorithm ultimately sets q(v, D) = dvv + p(v, D) − w(v) = p(v, D), as desired
            # ...
            # Otherwise, v is a leaf node in all minimum cost NWSTs that connect D ∪ {v}, in which case
            # the cost is q(v, D) = dvu + p(u, D) − w(u), where u is the first vertex of degree != 2 that
            # is encountered in the tree by starting a path from v.
            # ...
            # The term q(v, D) is the weight of a minimum NWST connecting D ∪ {v}

            for v in vertices:
                p[(v, D)] = math.inf
                for j in range(1, i):  # range is [1, i) or A:0 < |A| < |D|
                    for A in combinations(D, j):
                        D_minus_A = tuple([d for d in D if d not in A])
                        cost = q.get((v, A), math.inf) + q.get((v, D_minus_A), math.inf) - w(v)
                        if cost < p[(v, D)]:
                            p[(v, D)] = cost
                            p_split_info[(v, D)] = (v, A, D_minus_A)

            for v in vertices:
                for u in vertices:
                    cost = d_w[v][u] + p.get((u, D)) - w(u)
                    if cost < q.get((v, D), math.inf):
                        q[(v, D)] = cost
                        q_split_info[(v, D)] = p_split_info[(u, D)]

    D = tuple(terminals_minus_source)
    cost = q[(source, D)]

    return cost, D, q_split_info


def backtrack_solution(source: str | int, D, u: str | int, split_info, predecessors):
    r"""
    Rebuilds the Node-Weighted Steiner Tree connecting source to terminals D.

    Args:
        source: The source vertex (str or int).
        D: The set of terminals (tuple of vertices).
        u: The initial parent vertex for source (str or int).
        split_info: Dictionary mapping (v, D) to (u, A, D \ A) for leaf nodes or (v, A, D \ A) for interior nodes.
        predecessors: Dictionary mapping (v, u) to predecessor of v on the shortest path to u.

    Returns:
        Nodes forming the NWST.
    """

    def build_path(start, end):
        pred = predecessors[start]
        path = []
        while True:
            start = end
            if start is None:
                break
            path.append(start)
            end = pred[start]
        return path

    processed = set()
    nodes = set()

    def backtrack(v, D):
        if (v, D) in processed:
            return
        processed.add((v, D))

        # Base case: D is a single terminal
        if len(D) == 1:
            t = D[0]
            if v != t:
                if path := build_path(v, t):
                    nodes.update(path)
            return

        # v is either a leaf or interior node.
        u, A, D_minus_A = split_info.get((v, D))

        if u != v and u is not None and v is not None:
            # v is a leaf node, connect to u via a path
            if path := build_path(v, u):
                nodes.update(path)
            # Recurse on u with the same D
            backtrack(u, D)
        else:
            # v is an interior node, recurse on subsets A and D \ A
            backtrack(v, A)
            backtrack(v, D_minus_A)

    # Start backtracking from the source
    # Treat source as a leaf
    if path := build_path(source, u):
        nodes.update(path)
    backtrack(u, D)

    return sorted(list(nodes))


# Example usage
if __name__ == "__main__":
    edges_graph = {
        "a": {"b": 1},
        "b": {"a": 1, "c": 3, "i": 0},
        "c": {"b": 1, "d": 0},
        "d": {"c": 3, "e": 1, "k": 3},
        "e": {"d": 0, "f": 3},
        "f": {"e": 1, "g": 3, "l": 1},
        "g": {"f": 3, "h": 2, "n": 2},
        "h": {"g": 3, "i": 0},
        "i": {"b": 1, "h": 2, "j": 2},
        "j": {"i": 0, "k": 3},
        "k": {"d": 0, "j": 2, "l": 1, "n": 2},
        "l": {"f": 3, "k": 3, "m": 1},
        "m": {"l": 1},
        "n": {"g": 3, "k": 3},
    }
    graph = defaultdict(dict)
    vertices = graph.keys()

    # Use zero edge weights for pure node based calculations.
    for u in edges_graph:
        for v, weight in edges_graph[u].items():
            graph[u][v] = weight

    # fmt:off
    # vertex_weights = {"a": 1, "b": 1, "c": 3, "d": 0, "e": 1, "f": 3, "g": 3, "h": 2, "i": 0, "j": 2, "k": 3, "l": 1, "m": 1, "n": 2}
    vertex_weights = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0, "g": 0, "h": 0, "i": 0, "j": 0, "k": 0, "l": 0, "m": 0, "n": 0}
    # fmt:on

    distances = {}
    predecessors = {}
    for v in vertices:
        distances[v], predecessors[v] = DW_dijkstra(graph, vertex_weights, v, vertices)

    terminals = {"a", "d", "i", "m"}

    for source in terminals:
        start_time = time.perf_counter_ns()
        cost, D, q_split_info = DW_nwst(distances, vertex_weights, terminals, source)
        end_time = time.perf_counter_ns()
        execution_time_ms = (end_time - start_time) / 1_000_000

        u, _, _ = q_split_info[(source, D)]
        nodes = backtrack_solution(source, D, u, q_split_info, predecessors)

        print()
        print(f"Calculated: source: {source}, terminal_subset: {D}, cost: {cost}, u: {u}")
        print(f"  Calculated nodes: {nodes}")
        print("   Expected: nodes: ['a', 'b', 'd', 'i', 'j', 'k', 'l', 'm'], cost: 9")
        print(f"    Execution time: {execution_time_ms:.3f} ms")
        print("===================================================")
