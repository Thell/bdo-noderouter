# api_common.py

from joblib import Memory

from enum import IntEnum
from typing import TypedDict
import sys

from bidict import bidict
from loguru import logger
import rustworkx as rx

import data_store as ds

memory = Memory(location=".cache", verbose=0)

# Constants
GREAT_OCEAN_TERRITORY = 5
OQUILLAS_EYE_KEY = 1727
SUPER_ROOT = 99999


class NodeType(IntEnum):
    normal = 0
    village = 1
    city = 2
    gate = 3
    farm = 4
    trade = 5
    collect = 6
    quarry = 7
    logging = 8
    dangerous = 9
    finance = 10
    fish_trap = 11
    minor_finance = 12
    monopoly_farm = 13
    craft = 14
    excavation = 15
    count = 16


class ResultDict(TypedDict):
    solution_graph: rx.PyDiGraph
    objective_value: int
    duration: float


def set_logger(config: dict):
    log_level = config.get("logger", {}).get("level", "INFO")
    log_format = config.get("logger", {}).get("format", "<level>{message}</level>")
    logger.remove()
    logger.add(sys.stdout, colorize=True, level=log_level, format=log_format)
    return logger


@memory.cache
def get_clean_exploration_data(config: dict):
    """Read exploration.json from data store and recursively:
    - Remove all entries with an empty link list
    - Remove elements from all link lists that are not in exploration
    - If config.exploration_data.valid_nodes is a non-empty list then only valid nodes are kept
    - If config.exploration_data.omit_great_ocean is true omits all non-routable ocean nodes.
    """
    logger.trace("get_clean_exploration_data")
    data = {int(k): v for k, v in ds.read_json("exploration.json").items()}

    valid_nodes = config.get("exploration_data", {}).get("valid_nodes", [])
    assert isinstance(valid_nodes, list)
    if valid_nodes:
        data = {k: v for k, v in data.items() if k in valid_nodes}

    omit_great_ocean = config.get("exploration_data", {}).get("omit_great_ocean", False)
    if omit_great_ocean:
        data = {
            k: v for k, v in data.items() if v["is_base_town"] or v["territory_key"] != GREAT_OCEAN_TERRITORY
        }

    # Recursively remove all non valid entries from link lists and
    # remove all nodes with an empty link list.
    while True:
        valid_keys = {k for k, v in data.items() if v["link_list"]}

        for v in data.values():
            v["link_list"] = [neighbor for neighbor in v["link_list"] if neighbor in valid_keys]

        new_data = {k: v for k, v in data.items() if v["link_list"]}
        if len(new_data) == len(data):
            break
        data = new_data

    # ds.write_json(ds.path().joinpath("clean_exploration.json").as_posix(), data)

    return data


def load_graph(filename: str) -> rx.PyGraph | rx.PyDiGraph:
    """
    Load graph nodes, node waypoint keys, edges and terminals from json file
    in the projects data/solutions directory and return them as a rustworkx graph.

    Naming convention should follow the form:
        ["IP"|"LP"]_{num_roots}_{num_terminals}_{num_danger}_{notation}

    where num_danger > 0 indicates super root (99999) is included in the graph.
    Notation should be a human readable note like 'subgraph_pruned' or a test number.
    """
    print(f"Loading graph solution from {filename}")

    data_path = ds.path().joinpath("solutions").joinpath(filename)
    data = ds.read_json(data_path.as_posix())

    graph_type = data["graph_type"]
    graph = rx.PyDiGraph() if graph_type == "directed" else rx.PyGraph()
    nodes = data["nodes"]
    edges = data["edges"]
    terminals = data["terminals"]

    exploration_data = get_clean_exploration_data({})

    for i, waypoint_key in nodes.items():
        index = graph.add_node(exploration_data[waypoint_key])
        assert int(i) == index

    graph.add_edges_from([(u, v, edge_data) for u, v, edge_data in edges])

    # Reset graph.attrs
    graph.attrs = {"terminals": terminals}

    terminal_sets = {r: [] for r in terminals.values()}
    for k, v in terminals.items():
        terminal_sets[v].append(k)
    graph.attrs["terminal_sets"] = terminal_sets

    graph.attrs["node_key_by_index"] = bidict({i: graph[i]["waypoint_key"] for i in graph.node_indices()})

    return graph


def save_graph(graph: rx.PyGraph | rx.PyDiGraph, terminals: dict[int, int], filename: str):
    """
    Save graph nodes, node waypoint keys, edges and terminals to json file
    in the projects data/solutions directory.

    Naming convention should follow the form:
        ["IP"|"LP"]_{num_roots}_{num_terminals}_{num_danger}_{notation}

    where num_danger > 0 indicates super root (99999) is included in the graph.
    Notation should be a human readable note like 'subgraph_pruned' or a test number.
    """

    graph_type = "directed" if isinstance(graph, rx.PyDiGraph) else "undirected"

    # We need to ensure contiguous IDs in case there have been deleted nodes in the graph.
    index_map = {old_index: new_index for new_index, old_index in enumerate(graph.node_indexes())}

    # Store nodes in order of their new indices but with the waypoint key from the old index
    nodes = {}
    for old_index, new_index in index_map.items():
        nodes[new_index] = graph[old_index]["waypoint_key"]

    # Store edges with mapped node indices
    edges = [(index_map[u], index_map[v], graph.get_edge_data(u, v)) for u, v in graph.edge_list()]

    graph_type = "directed" if isinstance(graph, rx.PyDiGraph) else "undirected"
    num_components = None
    if graph_type == "directed":
        num_components = len(rx.strongly_connected_components(graph))  # type: ignore
    elif graph_type == "undirected":
        num_components = len(rx.connected_components(graph))  # type: ignore

    data = {
        "graph_type": graph_type,
        "num_components": num_components,
        "nodes": nodes,
        "edges": edges,
        "terminals": terminals,
    }

    data_path = ds.path().joinpath("solutions").joinpath(filename)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_json(data_path.as_posix(), data)
    print(f"Saved graph solution to {data_path}")
