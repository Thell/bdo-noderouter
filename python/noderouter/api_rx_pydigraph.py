# api_rx_pydigraph.py

from loguru import logger
from rustworkx import PyDiGraph


def set_graph_terminal_sets_attribute(graph: PyDiGraph, terminals: dict[int, int]):
    """Sets the graph's "terminals" and "terminal_sets" attributes.

    NOTE: This is used on the graph being sent to the solver.
    """
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


def subgraph_stable(graph: PyDiGraph, indices: list[int] | set[int], inclusive: bool = True) -> PyDiGraph:
    """Shallow copies ref graph and deletes all nodes not in indices if inclusive is True.
    Generates a 1:1 indexed subgraph of ref_graph eliminating node mapping.

    SAFETY: Nodes added to this copy will NOT have 1:1 indexing with ref_graph.
    """
    sub_graph = graph.copy()
    if not inclusive:
        sub_graph.remove_nodes_from(indices)
    else:
        sub_graph.remove_nodes_from(set(sub_graph.node_indices()) - set(indices))
    return sub_graph
