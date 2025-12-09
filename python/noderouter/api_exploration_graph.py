# api_exploration_graph.py

from dataclasses import dataclass
from enum import StrEnum

import rustworkx as rx
from bidict import bidict
from loguru import logger

import data_store as ds
from api_common import get_clean_exploration_data, NodeType


PAIRING_NEAREST_N = 3  # number of nearest roots to consider


@dataclass
class PairingData:
    def __init__(self):
        config = ds.get_config("config")
        exploration_data = get_clean_exploration_data(config)
        graph = get_exploration_graph(exploration_data)

        self.exploration_data = exploration_data
        self.graph = graph

        tmp = get_all_pairs_path_lengths(graph)
        self.path_lengths = {
            (graph[s]["waypoint_key"], graph[d]["waypoint_key"]): v for (s, d), v in tmp.items()
        }

        self.cartesian_distances = get_all_pairs_cartesian_distances(exploration_data)

        self.territories = set(n["territory_key"] for n in exploration_data.values())
        self.capitals = {
            d["territory_key"]: n for n, d in exploration_data.items() if d["node_type"] == NodeType.city
        }

        self.towns = [n for n, d in exploration_data.items() if d.get("is_town", False)]
        self.worker_towns = [n for n, d in exploration_data.items() if d.get("is_worker_npc_town", False)]

        tmp = {territory: [] for territory in self.territories}
        for t in self.towns:
            tmp[exploration_data[t]["territory_key"]].append(t)
        self.territory_towns = tmp

    def path_length(self, u: int, v: int) -> int:
        return self.path_lengths[(u, v)]

    def cartesian_distance(self, u: int, v: int) -> int:
        return self.cartesian_distances[(u, v)]

    def towns_in_territory(self, territory: int) -> list[int]:
        return self.territory_towns[territory]


PAIRING_DATA: PairingData  # Instantiated at the bottom of this file


class RootPairingType(StrEnum):
    # Capital-based
    capital = "capital"
    nearest_capital = "nearest_capital"
    cheapest_capital = "cheapest_capital"
    territory_capital = "territory_capital"

    # Town-based
    town = "town"
    nearest_town = "nearest_town"
    cheapest_town = "cheapest_town"
    nearest_town_in_territory = "nearest_town_in_territory"
    cheapest_town_in_territory = "cheapest_town_in_territory"

    # Randomized (unconstrained)
    random_capital = "random_capital"
    random_town = "random_town"
    random_any_root = "random_any_root"
    random_terminal_to_terminal = "random_terminal_to_terminal"

    # Randomized (top-N constrained)
    random_n_nearest_town = "random_n_nearest_town"
    random_n_cheapest_town = "random_n_cheapest_town"
    random_n_nearest_town_in_territory = "random_n_nearest_town_in_territory"
    random_n_cheapest_town_in_territory = "random_n_cheapest_town_in_territory"

    def candidates(self, terminal: dict) -> list[int]:
        """
        Return candidate root IDs for the given terminal under this pairing type.
        """
        match self:
            # Capital-based
            case RootPairingType.capital:
                return list(PAIRING_DATA.capitals.values())

            case RootPairingType.nearest_capital:
                nearest = min(
                    PAIRING_DATA.capitals.values(),
                    key=lambda c: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], c),
                )
                return [nearest]

            case RootPairingType.cheapest_capital:
                cheapest = min(
                    PAIRING_DATA.capitals.values(),
                    key=lambda c: PAIRING_DATA.path_length(terminal["waypoint_key"], c),
                )
                return [cheapest]

            case RootPairingType.territory_capital:
                # territory capitals are pre‑marked in exploration_data
                territory_id = terminal["territory_key"]
                return [PAIRING_DATA.capitals[territory_id]]

            # Town-based
            case RootPairingType.town:
                return PAIRING_DATA.towns

            case RootPairingType.nearest_town:
                nearest = min(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return [nearest]

            case RootPairingType.cheapest_town:
                cheapest = min(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return [cheapest]

            case RootPairingType.nearest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                nearest = min(
                    towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return [nearest]

            case RootPairingType.cheapest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                cheapest = min(
                    towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return [cheapest]

            # Randomized (unconstrained)
            case RootPairingType.random_capital:
                return list(PAIRING_DATA.capitals.values())

            case RootPairingType.random_town:
                return PAIRING_DATA.towns

            case RootPairingType.random_any_root:
                return list(PAIRING_DATA.capitals.values()) + PAIRING_DATA.towns

            case RootPairingType.random_terminal_to_terminal:
                # stress-test: allow any node to be treated as a root
                return list(PAIRING_DATA.exploration_data.keys())

            # Randomized (top-N constrained)
            case RootPairingType.random_n_nearest_town:
                nearest_sorted = sorted(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_NEAREST_N]

            case RootPairingType.random_n_cheapest_town:
                cheapest_sorted = sorted(
                    PAIRING_DATA.towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_NEAREST_N]

            case RootPairingType.random_n_nearest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                nearest_sorted = sorted(
                    towns,
                    key=lambda t: PAIRING_DATA.cartesian_distance(terminal["waypoint_key"], t),
                )
                return nearest_sorted[:PAIRING_NEAREST_N]

            case RootPairingType.random_n_cheapest_town_in_territory:
                towns = PAIRING_DATA.towns_in_territory(terminal["territory_key"])
                cheapest_sorted = sorted(
                    towns,
                    key=lambda t: PAIRING_DATA.path_length(terminal["waypoint_key"], t),
                )
                return cheapest_sorted[:PAIRING_NEAREST_N]

            case _:
                raise ValueError(f"Unknown pairing type: {self}")


def exploration_graph_nw(data: dict, directed: bool = False) -> rx.PyGraph | rx.PyDiGraph:
    """Generate and return a node weighted PyGraph graph from 'exploration.json'.
    The returned graph will have an attribute [node_key_by_index] containing all
    node indices and exploration keys.

    Indices and keys of the node key map are integer values.
    """
    if directed:
        logger.debug("Generating node weighted directed graph...")
    else:
        logger.debug("Generating node weighted undirected graph...")

    graph = rx.PyGraph(multigraph=False)

    # Map from exploration key to PyGraph node index (or inverse)
    node_key_by_index = bidict({i: k for k, i in zip(data.keys(), graph.add_nodes_from(data.values()))})

    # Add unweighted undirected edge from each node to its neighbors.
    # NOTE: If edges are added for both (i,j) and (j,i) then they
    #       are distinct edges that are both undirected.
    for i in graph.node_indices():
        neighbors = graph[i]["link_list"]
        for j_key in neighbors:
            j = node_key_by_index.inv[j_key]
            if not graph.has_edge(i, j):
                graph.add_edge(i, j, None)

    if directed:
        graph = graph.to_directed()
    graph.attrs = {"node_key_by_index": node_key_by_index}

    logger.debug(f"  generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")
    return graph


def exploration_graph_ew_directed(data) -> rx.PyGraph | rx.PyDiGraph:
    """Generate and return an edge weighted directed PyDiGraph from 'exploration.json'
    The returned graph will have an attribute [node_key_by_index] containing all
    node indices and exploration keys.

    Edge weighting is done by the process of splitting each node into two nodes, an _in
    and _out node for each original node respectively. All incoming arcs to the original
    node go to the 'in' node and all outgoing arcs come from the 'out' node.

    The payload of the connecting edge is the exploration data for the original node.
    All other edges have a None payload.

    Indices of the node key map are integer values and keys are strings consisting of the
    original node's waypoint_key attribute with a suffix of '_in' or '_out'.
    """
    logger.debug("Generating edge weighted directed graph...")

    graph = rx.PyDiGraph(multigraph=False)
    node_key_by_index: bidict[int, str] = bidict({})

    def split_exploration_node(exploration_node: dict):
        waypoint_key = exploration_node["waypoint_key"]

        in_key = f"{waypoint_key}_in"
        out_key = f"{waypoint_key}_out"
        if in_key in node_key_by_index.inv:
            assert out_key in node_key_by_index.inv
            return
        else:
            assert out_key not in node_key_by_index.inv

        tmp = exploration_node.copy()
        tmp["waypoint_key"] = in_key
        in_index = graph.add_node(tmp)

        tmp = exploration_node.copy()
        tmp["waypoint_key"] = out_key
        out_index = graph.add_node(tmp)

        node_key_by_index[in_index] = in_key
        node_key_by_index[out_index] = out_key
        graph.add_edge(in_index, out_index, exploration_node)

    # Split all exploration nodes.
    for exploration_node in data.values():
        split_exploration_node(exploration_node)

    # Add all outbound arcs using each node's "link_list".
    # Each link will go from the exploration nodes '_out' to the destination's '_in'
    # the node indices within the graph are obtained using the node_key_by_index map
    for exploration_key, exploration_data in data.items():
        out_key = f"{exploration_key}_out"
        out_index = node_key_by_index.inv[out_key]

        for destination_key in exploration_data["link_list"]:
            in_key = f"{destination_key}_in"
            in_index = node_key_by_index.inv[in_key]
            graph.add_edge(out_index, in_index, None)

    graph.attrs = {"node_key_by_index": node_key_by_index}
    print(f"  generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")
    return graph


def populate_edge_weights(graph: rx.PyGraph | rx.PyDiGraph):
    for u, v in graph.edge_list():
        graph.update_edge(u, v, {"need_exploration_point": graph[v]["need_exploration_point"]})


def has_edge_weights(graph: rx.PyGraph | rx.PyDiGraph) -> bool:
    has_edge_data = False
    for edge in graph.edges():
        if edge is not None and edge.get("need_exploration_point") is not None:
            has_edge_data = True
            break
    return has_edge_data


def get_all_pairs_shortest_paths(graph: rx.PyGraph | rx.PyDiGraph) -> dict[tuple[int, int], list[int]]:
    if not has_edge_weights(graph):
        populate_edge_weights(graph)
    shortest_paths = rx.all_pairs_dijkstra_shortest_paths(
        graph, edge_cost_fn=lambda edge_data: edge_data["need_exploration_point"]
    )
    shortest_paths = {
        (u, v): list(path)
        for u, paths_from_source in shortest_paths.items()
        for v, path in paths_from_source.items()
    }
    return shortest_paths


def get_all_pairs_path_lengths(graph: rx.PyGraph | rx.PyDiGraph) -> dict[tuple[int, int], int]:
    """Calculates and returns the node weighted shortest paths for all pairs."""
    if not has_edge_weights(graph):
        populate_edge_weights(graph)
    shortest_paths = get_all_pairs_shortest_paths(graph)
    return {
        key: sum(graph[w]["need_exploration_point"] for w in path) for key, path in shortest_paths.items()
    }


def get_all_pairs_cartesian_distances(exploration_data: dict) -> dict[tuple[int, int], int]:
    """Calculates and returns the cartesian distances for all pairs.
    Cordinates are taken from the exploration data's x, y, z positions where x and z are
    the horizontal and vertical positions from center of the map.
    """
    return {
        (u, v): abs(exploration_data[u]["position"]["x"] - exploration_data[v]["position"]["x"])
        + abs(exploration_data[u]["position"]["z"] - exploration_data[v]["position"]["z"])
        for u in exploration_data.keys()
        for v in exploration_data.keys()
    }


def get_exploration_graph(config: dict) -> rx.PyGraph | rx.PyDiGraph:
    """Returns a rustworkx graph.

    Arguments:
        directed: boolean (false: PyGraph, true: PyDigraph)
        edge_weighted: boolean (false: no weighting, true: see note)

    Note: When edge_weighted is true this implies directed is true. Nodes
    will be split into an _in and _out node with a single directed arc
    between them with a weight equal to the node's cost.
    """
    directed: bool = config.get("exploration_data", {}).get("directed", False)
    edge_weighted: bool = config.get("exploration_data", {}).get("edge_weighted", False)
    if not isinstance(directed, bool) or not isinstance(edge_weighted, bool):
        raise ValueError("directed and edge_weighted must be booleans")

    data = get_clean_exploration_data(config)

    match (directed, edge_weighted):
        case (False, False):
            graph = exploration_graph_nw(data, False)
        case (True, False):
            graph = exploration_graph_nw(data, True)
        case (False, True):
            graph = exploration_graph_ew_directed(data)
        case (True, True):
            graph = exploration_graph_ew_directed(data)
    return graph


def get_neighboring_territories(exploration_data: dict):
    """Processes exploration data to generate and return dict of `{territory: [neighbors]}`.
    Note: territory is included in neighbors

    If it is desired to omit the great ocean nodes then that should be done when creating
    the expploration data.
    """
    territories = set()
    territory_pairs = set()
    for node_data in exploration_data.values():
        node_territory = node_data["territory_key"]
        territories.add(node_territory)
        for neighbor in node_data["link_list"]:
            neighbor_data = exploration_data.get(neighbor, None)
            if neighbor_data:
                neighbor_territory = neighbor_data["territory_key"]
                territories.add(neighbor_territory)
                territory_pairs.add((node_territory, neighbor_territory))

    results = {t: set() for t in territories}
    for t1, t2 in territory_pairs:
        results[t1].add(t2)
        results[t2].add(t1)
    results = {t: sorted(v) for t, v in results.items()}
    return results


def get_neighboring_region_groups(config: dict) -> dict[int, list[int]]:
    """Processes exploration data to generate and return a dict of `{region_group: [neighbors]}`.
    Note: region_group is included in neighbors
    """
    exploration_data = get_clean_exploration_data(config)
    region_groups = set()
    region_group_pairs = set()
    for node_data in exploration_data.values():
        node_territory = node_data["territory_key"]
        node_region_group = node_data["region_group_key"]
        region_groups.add(node_territory)

        for neighbor in node_data["link_list"]:
            neighbor_data = exploration_data.get(neighbor, None)
            if neighbor_data:
                neighbor_region_group = neighbor_data["region_group_key"]
                region_groups.add(neighbor_region_group)
                region_group_pairs.add((node_region_group, neighbor_region_group))

    results = {t: set() for t in region_groups}
    for t1, t2 in region_group_pairs:
        results[t1].add(t2)
        results[t2].add(t1)
    results = {t: sorted(v) for t, v in results.items()}
    return results


def generate_region_group_neighbors(waypoint_key: int, config: dict):
    exploration_nodes = get_clean_exploration_data(config)
    waypoint_region_group = exploration_nodes[waypoint_key]["region_group_key"]

    region_group_neighbors = get_neighboring_region_groups(config)
    neighbor_region_groups = region_group_neighbors[waypoint_region_group]

    neighbors = set()
    for node in exploration_nodes.values():
        if node["region_group_key"] in neighbor_region_groups:
            neighbors.add(node["waypoint_key"])
    neighbors = sorted(list(neighbors))
    return neighbors


def get_territory_root_sets(exploration_data: dict, territory_neighbors: dict):
    """Generate a dict of root nodes within a territory and its' neighbors."""
    territory_root_sets = {t: set() for t in territory_neighbors}

    for node_key, node_data in exploration_data.items():
        if not node_data["is_base_town"]:
            continue

        root_territory = node_data["territory_key"]
        if root_territory not in territory_neighbors:
            continue

        for n in territory_neighbors[root_territory]:
            territory_root_sets[n].add(node_key)

    territory_root_sets = {k: list(sorted(v)) for k, v in territory_root_sets.items()}
    return territory_root_sets


def generate_territory_root_sets(exploration_data: dict):
    territory_neighbors = get_neighboring_territories(exploration_data)
    territory_root_sets = get_territory_root_sets(exploration_data, territory_neighbors)
    return territory_root_sets


def get_super_root(config: dict) -> dict:
    """Returns a base town exploration node with waypoint_key: 99999 and
    link_list consisting of all other exploration nodes with "is_base_town"
    attribute set. If valid_nodes is not empty super root's link_list will
    be filtered to the valid_nodes.

    This facilitates the connection of terminals through any potential root in the graph.

    NOTE: Adding super root to the graph breaks graph planarity!
    """
    data = get_clean_exploration_data(config)
    link_list = [k for k, v in data.items() if v["is_base_town"]]

    return {
        "waypoint_key": 99999,
        "region_key": 99999,
        "region_group_key": 99999,
        "territory_key": 99999,
        "character_key": 99999,
        "node_type": 1,
        "is_town": True,
        "is_base_town": True,
        "is_plantzone": False,
        "is_workerman_plantzone": False,
        "is_warehouse_town": False,
        "is_worker_npc_town": False,
        "need_exploration_point": 0,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "link_list": link_list,
        "worker_types": [],
        "region_houseinfo": {
            "has_rentable_lodging": False,
            "has_rentable_storage": False,
            "has_cashproduct_lodging": False,
            "has_cashproduct_storage": False,
        },
    }


# Module level pairing data
PAIRING_DATA = PairingData()
