# api_rx_pydigraph.py

from loguru import logger
import rustworkx as rx

import api_exploration_graph as exploration_api

SUPER_ROOT = 99999


def set_graph_terminal_sets_attribute(graph: rx.PyDiGraph, terminals: dict[int, int]):
    attrs = graph.attrs
    if "node_key_by_index" not in attrs:
        raise ValueError("'node_key_by_index' not in graph.attrs!")
    node_key_by_index = attrs["node_key_by_index"]

    if "terminals" in attrs or "terminal_sets" in attrs:
        logger.warning("'terminals' or 'terminal_sets' already in graph.attrs! Resetting...")

    terminal_sets = {node_key_by_index.inv[r_key]: [] for r_key in terminals.values()}
    for t_key, r_key in terminals.items():
        terminal_sets[node_key_by_index.inv[r_key]].append(node_key_by_index.inv[t_key])

    attrs["terminals"] = terminals
    attrs["terminal_sets"] = terminal_sets
    graph.attrs = attrs


def inject_super_root(config: dict, G: rx.PyDiGraph, flow_direction: str = "inbound"):
    """Injects the superroot node into the graph with either inbound, outbound,
    or undirected (both) arcs between the super root and all nodes in its "link_list".

    For reverse cumulative flow models this should be 'inbound' to signify flow from
    terminals to the roots.
    """
    logger.debug("Injecting super root...")
    if flow_direction not in ["inbound", "outbound", "undirected", "none"]:
        raise ValueError(f"Invalid flow_direction: {flow_direction}")
    if "node_key_by_index" not in G.attrs:
        raise ValueError("'node_key_by_index' not in graph.attrs!")

    node_key_by_index = G.attrs["node_key_by_index"]

    if SUPER_ROOT in node_key_by_index.inv:
        logger.warning("  super root already exists in graph! Skipping injection...")
        return

    super_root = exploration_api.get_super_root(config)
    super_root_index = G.add_node(super_root)
    assert super_root_index not in node_key_by_index
    node_key_by_index[super_root_index] = SUPER_ROOT

    if flow_direction != "none":
        logger.debug(f"  linking {flow_direction} to {super_root['link_list']}")
        if flow_direction in ["inbound", "undirected"]:
            for node_key in super_root["link_list"]:
                node_index = node_key_by_index.inv[node_key]
                G.add_edge(node_index, super_root_index, super_root)
        if flow_direction in ["outbound", "undirected"]:
            for node_key in super_root["link_list"]:
                node_index = node_key_by_index.inv[node_key]
                G.add_edge(super_root_index, node_index, super_root)

    G.attrs["node_key_by_index"] = node_key_by_index
    logger.debug(f"  injected at index {super_root_index}...")


def subgraph_stable(
    graph: rx.PyGraph | rx.PyDiGraph, indices: list[int] | set[int], inclusive: bool = True
) -> rx.PyGraph | rx.PyDiGraph:
    """Copies ref graph and deletes all nodes not in indices if inclusive is True.

    This generates a subgraph that has 1:1 index matching with ref_graph which eliminates
    the need for node_map. If any nodes are added to this copy those nodes will **NOT**
    have the same indices as they would in ref_graph.
    """
    dup_graph = graph.copy()
    if not inclusive:
        dup_graph.remove_nodes_from(indices)
    else:
        dup_graph.remove_nodes_from(set(dup_graph.node_indices()) - set(indices))
    return dup_graph
