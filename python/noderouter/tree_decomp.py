import math
from collections import Counter

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in, treewidth_min_degree
from pyvis.network import Network


class BDOStructuralAnalyzer:
    def __init__(self, nx_graph, data):
        if nx_graph.is_directed():
            raise ValueError("Structural analysis requires an undirected graph.")
        self.data = data
        self.full_graph = nx_graph
        self.reduced_graph = None
        self.core_graph = None
        self.skeleton_graph = None

    def analyze(self):
        print("--- Full Graph Analysis ---")
        self._profile_graph(self.full_graph)

        print("\n--- Reduced Graph (D1 Removal) ---")
        self.reduced_graph = self._get_reduced_graph(self.full_graph)
        self._profile_graph(self.reduced_graph)

        print("\n--- Core Graph (Series Reduction) ---")
        self.core_graph = self._get_essential_core(self.reduced_graph)
        self._profile_graph(self.core_graph, run_exact=True)

        print("\n--- Graph Skeleton (Y-Delta Reduction) ---")
        self.skeleton_graph = self._get_structural_skeleton(self.core_graph)
        self._profile_graph(self.skeleton_graph, run_exact=True)

    def _profile_graph(self, G, run_exact=False):
        width_stats = self._get_tight_upper_bound(G)
        upper_bound, decomposition = treewidth_min_fill_in(G)
        deg_counter = Counter(dict(G.degree()).values())
        print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        print(f"Heuristics: {width_stats}")
        print(f"Degree Distribution: {deg_counter}")
        print(f"Heuristic Treewidth (Min-Fill): {upper_bound}")
        print(f"Decomposition Bags: {len(decomposition.nodes())}")
        if run_exact:
            exact = self._get_exact_treewidth(G)
            print(f"Exact Treewidth: {exact}")

    def _get_reduced_graph(self, G):
        reduced = G.copy()
        base_towns = self.data.capitals
        while True:
            to_remove = [n for n, deg in reduced.degree() if deg < 2 and n not in base_towns]
            if not to_remove:
                break
            reduced.remove_nodes_from(to_remove)
        return reduced

    def _get_essential_core(self, G):
        core = G.copy()
        core.remove_edges_from(nx.selfloop_edges(core))
        while True:
            target_node = next((n for n, deg in core.degree() if deg == 2), None)
            if target_node is None:
                break
            neighbors = list(core.neighbors(target_node))
            u, w = neighbors[0], neighbors[1]
            if u != w and not core.has_edge(u, w):
                core.add_edge(u, w)
            core.remove_node(target_node)
        return core

    def _get_structural_skeleton(self, G):
        skel = G.copy()
        skel.remove_edges_from(nx.selfloop_edges(skel))
        while True:
            to_remove = [n for n, deg in skel.degree() if deg < 2]
            if to_remove:
                skel.remove_nodes_from(to_remove)
                continue
            t_d2 = next((n for n, deg in skel.degree() if deg == 2), None)
            if t_d2:
                u, w = list(skel.neighbors(t_d2))
                if u != w and not skel.has_edge(u, w):
                    skel.add_edge(u, w)
                skel.remove_node(t_d2)
                continue
            t_d3 = next((n for n, deg in skel.degree() if deg == 3), None)
            if t_d3:
                u, v, w = list(skel.neighbors(t_d3))
                for edge in [(u, v), (v, w), (u, w)]:
                    if edge[0] != edge[1] and not skel.has_edge(*edge):
                        skel.add_edge(*edge)
                skel.remove_node(t_d3)
                continue
            break
        return skel

    def _get_tight_upper_bound(self, G):
        res_fill, _ = treewidth_min_fill_in(G)
        res_degree, _ = treewidth_min_degree(G)
        return {"min_fill": res_fill, "min_degree": res_degree, "best": min(res_fill, res_degree)}

    def _get_exact_treewidth(self, G):
        memo = {}

        def solve(nodes):
            nodes = frozenset(nodes)
            if len(nodes) <= 1:
                return 0
            if nodes in memo:
                return memo[nodes]
            sub = G.subgraph(nodes)
            res = float("inf")
            ordered_nodes = sorted(nodes, key=lambda n: sub.degree(n))
            for n in ordered_nodes[:5]:
                neighbors = list(sub.neighbors(n))
                cost = len(neighbors)
                current_width = max(cost, solve(set(nodes) - {n}))
                res = min(res, current_width)
                if res <= 1:
                    break
            memo[nodes] = res
            return res

        lb = len(nx.approximation.max_clique(G)) - 1
        ub, _ = treewidth_min_fill_in(G)
        return lb if lb == ub else ub


class NiceNode:
    def __init__(self, bag, node_type, children=None, aux=None):
        self.bag = frozenset(bag)
        self.node_type = node_type
        self.children = children if children else []
        self.aux = aux
        self.table = {}


class BDONiceTreeGenerator:
    def __init__(self, T, root_bag):
        self.T = T
        self.root_bag = frozenset(root_bag)

    def generate(self):
        directed_T = nx.bfs_tree(self.T, self.root_bag)

        def transform(u_bag):
            children_bags = list(directed_T.successors(u_bag))
            curr_bag = frozenset(u_bag)
            if len(children_bags) > 1:
                kids = [self._align(transform(v), curr_bag) for v in children_bags]
                res = kids[0]
                for i in range(1, len(kids)):
                    res = NiceNode(curr_bag, "join", [res, kids[i]])
                return res
            if not children_bags:
                node = NiceNode(frozenset(), "leaf")
                for n in curr_bag:
                    node = NiceNode(node.bag | {n}, "introduce", [node], n)
                return node
            return self._align(transform(children_bags[0]), curr_bag)

        return transform(self.root_bag)

    def _align(self, child_node, target_bag):
        curr = child_node
        for n in curr.bag - target_bag:
            curr = NiceNode(curr.bag - {n}, "forget", [curr], n)
        for n in target_bag - curr.bag:
            curr = NiceNode(target_bag, "introduce", [curr], n)
        return curr


class BDONiceTreeManager:
    def __init__(self, nx_graph, exploration_data):
        self.full_graph = nx_graph
        self.data = exploration_data
        self.key_by_index = self.data.graph.attrs["node_key_by_index"]
        self.index_by_key = self.key_by_index.inv
        self.root = None
        self.colors = {
            "leaf": "#757575",
            "introduce": "#4caf50",
            "forget": "#f44336",
            "join": "#2196f3",
            "trace_a": "#ffeb3b",
            "trace_b": "#00bcd4",
            "overlap": "#ff9800",
        }

    def generate_nice_tree(self):
        reduced = self._get_reduced_graph(self.full_graph)
        _, T = treewidth_min_fill_in(reduced)
        root_bag = list(T.nodes())[0]
        self.root = self._transform_to_nice(T, root_bag)
        return self.root

    def _get_reduced_graph(self, G):
        reduced = G.copy()
        base_towns = self.data.capitals
        while True:
            to_remove = [n for n, deg in reduced.degree() if deg < 2 and n not in base_towns]
            if not to_remove:
                break
            reduced.remove_nodes_from(to_remove)
        return reduced

    def _transform_to_nice(self, T, root_bag):
        directed_T = nx.bfs_tree(T, root_bag)

        def transform(u_bag):
            children = list(directed_T.successors(u_bag))
            curr = frozenset(u_bag)
            if len(children) > 1:
                kids = [self._align(transform(v), curr) for v in children]
                res = kids[0]
                for i in range(1, len(kids)):
                    res = NiceNode(curr, "join", [res, kids[i]])
                return res
            if not children:
                node = NiceNode(frozenset(), "leaf")
                for n in curr:
                    node = NiceNode(node.bag | {n}, "introduce", [node], n)
                return node
            return self._align(transform(children[0]), curr)

        def align(node, target):
            curr = node
            for n in curr.bag - target:
                curr = NiceNode(curr.bag - {n}, "forget", [curr], n)
            for n in target - curr.bag:
                curr = NiceNode(target, "introduce", [curr], n)
            return curr

        return transform(root_bag)

    def _align(self, node, target):
        curr = node
        for n in curr.bag - target:
            curr = NiceNode(curr.bag - {n}, "forget", [curr], n)
        for n in target - curr.bag:
            curr = NiceNode(target, "introduce", [curr], n)
        return curr

    def visualize(self, filename="ntd_output.html", layout="hierarchical", trace_keys=None):
        net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white", directed=True)

        # Strictly integer-based lookup
        id_a = self.index_by_key[int(trace_keys[0])] if trace_keys and len(trace_keys) > 0 else None
        id_b = self.index_by_key[int(trace_keys[1])] if trace_keys and len(trace_keys) > 1 else None

        node_metadata = self._get_node_metadata() if layout == "radial" else {}
        overlap_count = 0

        def traverse(node, start_angle=0, end_angle=2 * math.pi):
            nonlocal overlap_count
            node_id = id(node)
            has_a = id_a in node.bag if id_a is not None else False
            has_b = id_b in node.bag if id_b is not None else False

            if has_a and has_b:
                overlap_count += 1
                color, size = self.colors["overlap"], 35
            elif has_a:
                color, size = self.colors["trace_a"], 30
            elif has_b:
                color, size = self.colors["trace_b"], 30
            else:
                color, size = self.colors.get(node.node_type, "#ffffff"), 12

            aux_key = self.key_by_index[node.aux] if node.aux is not None else None
            label = f"{node.node_type.upper()}" + (f"\n({aux_key})" if aux_key is not None else "")

            pos = {}
            if layout == "radial":
                meta = node_metadata[node_id]
                rad = meta["depth"] * 250
                ang = (start_angle + end_angle) / 2
                pos = {"x": rad * math.cos(ang), "y": rad * math.sin(ang), "physics": False}

            sorted_bag = sorted([self.key_by_index[n] for n in node.bag])
            title_text = f"Bag: {sorted_bag}"

            net.add_node(
                node_id,
                label=label,
                color=color,
                size=size,
                title=title_text,
                **pos,
            )

            curr_ang = start_angle
            for child in node.children:
                if layout == "radial":
                    c_meta = node_metadata[id(child)]
                    sector = (end_angle - start_angle) * (
                        c_meta["leaf_count"] / node_metadata[node_id]["leaf_count"]
                    )
                    traverse(child, curr_ang, curr_ang + sector)
                    curr_ang += sector
                else:
                    traverse(child)
                net.add_edge(id(child), node_id, color=color if size > 12 else "#444444")

        traverse(self.root)
        net.set_options(self._get_layout_options(layout))
        net.save_graph(filename)

        if trace_keys:
            print(f"Trace {trace_keys}: {overlap_count} overlapping bags identified and saved to {filename}.")

    def _get_node_metadata(self):
        metadata = {}

        def walk(node, depth):
            count = 1 if not node.children else sum(walk(c, depth + 1) for c in node.children)
            metadata[id(node)] = {"depth": depth, "leaf_count": count}
            return count

        walk(self.root, 0)
        return metadata

    def _get_layout_options(self, layout):
        if layout == "radial":
            return '{"physics": {"enabled": false}, "interaction": {"zoomView": true}}'
        return """{"layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "hubsize"}}, "physics": {"enabled": false}}"""

    def get_bottleneck_nodes(self, width=10, as_waypoints=False):
        """
        Scans the tree and returns the set of integer indices present
        in all bags that hit the maximum treewidth.
        """
        bottleneck_indices = set()

        def traverse(node):
            if len(node.bag) == width:
                bottleneck_indices.update(node.bag)
            for child in node.children:
                traverse(child)

        traverse(self.root)

        if as_waypoints:
            return [self.key_by_index[i] for i in bottleneck_indices]
        return bottleneck_indices

    def calculate_bottleneck_weights(self, target_key, as_waypoints=False):
        """
        Returns a dictionary mapping {bottleneck_id: frequency_of_cooccurrence}
        """
        target_id = self.index_by_key[int(target_key)]
        bottleneck_ids = self.get_bottleneck_nodes()

        weights = Counter()

        def walk(node):
            if target_id in node.bag:
                intersection = node.bag.intersection(bottleneck_ids)
                for b_id in intersection:
                    weights[b_id] += 1
            for child in node.children:
                walk(child)

        walk(self.root)
        weights = dict(sorted(weights.items()))
        if as_waypoints:
            return {self.key_by_index[b_id]: w for b_id, w in weights.items()}
        return weights

    def export_bottleneck_folium_data(self, target_keys):
        """
        Exports a dictionary structured for Folium mapping:
        { target_key: { bottleneck_id: { 'pct': float, 'weight': int }, 'total': int } }
        """
        export_data = {}
        for key in target_keys:
            weights = self.calculate_bottleneck_weights(key)
            total = sum(weights.values())

            export_data[key] = {
                "total_weight": total,
                "influences": {
                    b_id: {"percentage": round((w / total) * 100, 2), "weight": w}
                    for b_id, w in weights.items()
                },
            }
        return export_data


def find_bottleneck_nodes(nice_root, exploration_data):
    """Identifies the specific node_keys involved in the width-10 bags."""
    bottleneck_bags = []

    def traverse(node):
        if len(node.bag) == 10:
            bottleneck_bags.append(node.bag)
        for child in node.children:
            traverse(child)

    traverse(nice_root)

    # Flatten and get unique node indices
    all_indices = set().union(*bottleneck_bags)
    node_key_by_index = exploration_data.graph.attrs["node_key_by_index"]

    unique_keys = sorted([str(node_key_by_index[i]) for i in all_indices])

    print("\n" + "!" * 40)
    print(f"THE WIDTH-10 BOTTLENECK ({len(unique_keys)} Unique Nodes)")
    print("!" * 40)
    print(", ".join(unique_keys))
    print("!" * 40)

    return unique_keys


def print_final_decomposition_report(nice_root):
    widths = []
    types = Counter()

    def traverse(node):
        widths.append(len(node.bag))
        types[node.node_type] += 1
        for c in node.children:
            traverse(c)

    traverse(nice_root)
    width_dist = Counter(widths)

    print("\n" + "=" * 40)
    print("FINAL NICE TREE STRUCTURAL REPORT")
    print("=" * 40)
    print(f"{'Node Type':<15} | {'Count':<10}")
    print("-" * 30)
    for t, count in types.items():
        print(f"{t.capitalize():<15} | {count:<10}")

    print("\n--- Bag Width Distribution ---")
    print(f"{'Width (k+1)':<15} | {'Occurrence':<10}")
    print("-" * 30)
    for w in sorted(width_dist.keys()):
        print(f"{w:<15} | {width_dist[w]:<10}")

    print("=" * 40)


if __name__ == "__main__":
    from api_exploration_data import get_exploration_data

    print("Initializing Exploration Data...")
    exploration_data = get_exploration_data()
    node_key_by_index = exploration_data.graph.attrs["node_key_by_index"]

    # 1. Prepare Graph
    rx_full = exploration_data.graph.copy()
    rx_undirected = rx_full.to_undirected(multigraph=False)
    nx_G = nx.Graph()
    for u, v in rx_undirected.edge_list():
        nx_G.add_edge(u, v)

    # 2. Structural Analysis
    analyzer = BDOStructuralAnalyzer(nx_G, exploration_data)
    analyzer.analyze()

    # 3. Manager Initialization & Generation
    manager = BDONiceTreeManager(nx_G, exploration_data)
    nice_root = manager.generate_nice_tree()

    # 4. Reports
    # (Using your existing standalone report functions)
    find_bottleneck_nodes(nice_root, exploration_data)
    print_final_decomposition_report(nice_root)

    # 5. Visualization Generation

    # Standard Hierarchical View (No Trace)
    manager.visualize(filename="bdo_nice_tree_interactive.html", layout="hierarchical")

    # Single Trace: Calpheon (601) - Hierarchical
    manager.visualize(filename="nice_tree_trace_601.html", layout="hierarchical", trace_keys=[601])

    # Single Trace: Calpheon (601) - Radial
    manager.visualize(filename="bdo_nice_tree_radial_interactive.html", layout="radial", trace_keys=[601])

    # Dual Trace: Velia (1) vs Port Epheria (604) - Hierarchical
    manager.visualize(filename="nice_tree_dual_trace_1_604.html", layout="hierarchical", trace_keys=[1, 604])

    # Sample bottleneck weights
    target_id = 375
    weights = manager.calculate_bottleneck_weights(target_id)
    weights = {node_key_by_index[k]: v for k, v in weights.items()}
    print("\n" + "=" * 40)
    print("BOTTLENECK WEIGHTS")
    print(f"target: {target_id} | weights: {weights}")
