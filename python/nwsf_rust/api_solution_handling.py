from bidict import bidict
import highspy
from loguru import logger
import pyoptinterface as poi
from pyoptinterface.highs import Model
import rustworkx as rx

import api_rx_pydigraph as rx_api
import api_common as common_api


SUPER_ROOT = common_api.SUPER_ROOT


def validate_solution(solution_graph: rx.PyDiGraph):
    """Ensures the shortest paths between each root and terminal within territory regions
    of the subgraph components.
    """
    logger.info(
        "  ensuring solution contains shortest paths between roots and terminals per neighboring territories..."
    )

    has_error = False
    node_key_by_index = solution_graph.attrs["node_key_by_index"]
    terminal_sets = solution_graph.attrs["terminal_sets"]

    for r_index, terminal_set in terminal_sets.items():
        r_key = node_key_by_index[r_index]
        for t_index in terminal_set:
            t_key = node_key_by_index[t_index]
            # Some models use single arcs that are inbound and some that are outbound
            # for the super root so use the base towns which have anti-parallel arcs.
            if r_key == SUPER_ROOT:
                has_path = any(
                    rx.has_path(solution_graph, tmp_r, t_index)
                    for tmp_r in solution_graph.node_indices()
                    if solution_graph[tmp_r]["is_base_town"] and tmp_r != SUPER_ROOT
                )
            else:
                has_path = rx.has_path(solution_graph, r_index, t_index)
            if not has_path:
                logger.error(f"  no path between root {r_index} ({r_key}) and terminal {t_index} ({t_key})!")
                has_error = True

    if has_error:
        raise ValueError("Something is wrong with the core formula or solution extraction!")


def cleanup_solution(solution_graph: rx.PyDiGraph):
    """Cleanup solution: remove nodes not used in any path of terminal_sets."""
    logger.info("Cleaning solution...")

    terminal_sets = solution_graph.attrs["terminal_sets"]
    node_key_by_index = solution_graph.attrs["node_key_by_index"]

    super_root_index = node_key_by_index.inv.get(SUPER_ROOT, None)

    isolates = list(rx.isolates(solution_graph))
    if isolates:
        logger.info(f"  removing {len(isolates)} isolates... {[node_key_by_index[i] for i in isolates]}")
        logger.debug(f"  isolates: {[node_key_by_index[i] for i in isolates]}")
        total_cost = sum([solution_graph[i]["need_exploration_point"] for i in rx.isolates(solution_graph)])
        if total_cost > 0:
            logger.error(f"  isolates cost: {total_cost}")
        else:
            logger.debug(f"  isolates cost: {total_cost}")
        solution_graph.remove_nodes_from(isolates)

    # Preprocess arcs to include c(e) + w(v) in a new attribute
    for edge_index, (_, v, edge_data) in solution_graph.edge_index_map().items():
        if edge_data is None:
            edge_data = {}

        node_weight = solution_graph[v].get("need_exploration_point", 0.0)
        new_cost = edge_data.get("cost", 0.0) + node_weight
        edge_data["dijkstra_cost"] = new_cost
        solution_graph.update_edge_by_index(edge_index, edge_data)

    node_indices_to_remove = set(solution_graph.node_indices())
    for r_index, terminal_set in terminal_sets.items():
        r_key = node_key_by_index[r_index]
        for t_index in terminal_set:
            if r_key == SUPER_ROOT:
                for potential_root_index in solution_graph.node_indices():
                    if (
                        potential_root_index != SUPER_ROOT
                        and solution_graph[potential_root_index]["is_base_town"]
                    ):
                        if not rx.has_path(solution_graph, potential_root_index, t_index):
                            continue

                        paths = rx.dijkstra_shortest_paths(
                            solution_graph,
                            potential_root_index,
                            t_index,
                            weight_fn=lambda edge_data: edge_data["dijkstra_cost"],
                        )
                        for index in paths[t_index]:
                            node_indices_to_remove.discard(index)
            else:
                if not rx.has_path(solution_graph, r_index, t_index):
                    logger.error(
                        "  no path between root {r_key} ({r_index}) and terminal {t_key} ({t_index})!"
                    )
                    continue
                paths = rx.dijkstra_shortest_paths(
                    solution_graph, r_index, t_index, weight_fn=lambda edge_data: edge_data["dijkstra_cost"]
                )
                for index in paths[t_index]:
                    node_indices_to_remove.discard(index)

    if node_indices_to_remove:
        logger.info(f"  removing {len(node_indices_to_remove)} nodes not used in shortest paths...")
        logger.debug(f"  unused nodes: {[node_key_by_index[i] for i in node_indices_to_remove]}")
        total_cost = sum([solution_graph[i]["need_exploration_point"] for i in node_indices_to_remove])
        if total_cost > 0:
            logger.error(f"  unused nodes cost: {total_cost}")
        else:
            logger.debug(f"  unused nodes cost: {total_cost}")
        solution_graph.remove_nodes_from(node_indices_to_remove)
    else:
        logger.info("  no unused nodes to remove from solution...")

    # Fix node_key_from_index after removals
    if len(node_indices_to_remove) > 0:
        solution_graph.attrs["node_key_by_index"] = bidict({
            i: solution_graph[i]["waypoint_key"] for i in solution_graph.node_indices()
        })


def extract_solution_from_x_vars(model: Model, vars: dict, G: rx.PyDiGraph, config: dict) -> rx.PyDiGraph:
    """Create a subgraph from the graph consisting of the node x vars from the solved model."""
    logger.info("Extracting solution from x vars...")

    x_vars = vars["x"]

    solution_nodes = []
    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status == poi.TerminationStatusCode.OPTIMAL:
        solution_nodes = [i for i in G.node_indices() if round(model.get_value(x_vars[i])) == 1]
    else:
        logger.error(f"ERROR: Non optimal result status of {status}!")
        raise ValueError("Non optimal result status!")

    solution_graph = G.subgraph(solution_nodes, preserve_attrs=False)
    if len(solution_graph.node_indices()) == 0:
        logger.warning("Result is an empty solution.")
        return solution_graph

    node_key_by_index = bidict({i: solution_graph[i]["waypoint_key"] for i in solution_graph.node_indices()})
    solution_graph.attrs = {}
    solution_graph.attrs["node_key_by_index"] = node_key_by_index

    rx_api.set_graph_terminal_sets_attribute(solution_graph, G.attrs["terminals"])

    solution_options = config.get("solution", {})
    # Validation is done prior to cleanup becuase if SUPER_ROOT is present then it is needed for
    # validation.
    if solution_options.get("validate", False):
        validate_solution(solution_graph)
    if solution_options.get("cleanup", False):
        cleanup_solution(solution_graph)

    print(f"Solution: {[n['waypoint_key'] for n in solution_graph.nodes()]}")
    print(f"Solution Cost: {sum(n['need_exploration_point'] for n in solution_graph.nodes())}")
    return solution_graph


def extract_solution_from_x_vars_highspy(
    model: highspy.Highs, vars: dict, G: rx.PyDiGraph, config: dict
) -> rx.PyDiGraph:
    """Create a subgraph from the graph consisting of the node x vars from the solved model."""
    logger.info("Extracting solution from x vars from highspy...")

    x_vars = vars["x"]

    solution_nodes = []
    status = model.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal or highspy.HighsModelStatus.kInterrupt:
        solution_nodes = [
            i for i in G.node_indices() if round(model.getSolution().col_value[x_vars[i].index]) == 1
        ]
    else:
        logger.error(f"ERROR: Non optimal result status of {status}!")
        raise ValueError("Non optimal result status!")

    solution_graph = G.subgraph(solution_nodes, preserve_attrs=False)
    if len(solution_graph.node_indices()) == 0:
        logger.warning("Result is an empty solution.")
        return solution_graph

    node_key_by_index = bidict({i: solution_graph[i]["waypoint_key"] for i in solution_graph.node_indices()})
    solution_graph.attrs = {}
    solution_graph.attrs["node_key_by_index"] = node_key_by_index

    rx_api.set_graph_terminal_sets_attribute(solution_graph, G.attrs["terminals"])

    solution_options = config.get("solution", {})
    # Validation is done prior to cleanup becuase if SUPER_ROOT is present then it is needed for
    # validation.
    if solution_options.get("validate", False):
        validate_solution(solution_graph)
    if solution_options.get("cleanup", False):
        cleanup_solution(solution_graph)

    if config.get("logger", {}).get("level", "INFO") in ["INFO", "DEBUG", "TRACE"]:
        logger.info(f"Solution: {[n['waypoint_key'] for n in solution_graph.nodes()]}")
        logger.info(f"Solution Cost: {sum(n['need_exploration_point'] for n in solution_graph.nodes())}")
    return solution_graph


def extract_solution_from_xy_vars(
    model: Model, vars: dict, G: rx.PyDiGraph, G_prime: rx.PyDiGraph, **kwargs
) -> rx.PyDiGraph:
    """Create a subgraph from the graph consisting of the node x vars, edge y vars from the solved model.

    NOTE: G_prime contains the solution to the possibly reduced problem and G contains the original graph.
    Any selected node in G_prime must be translated to G because the ancestors will not exist in the
    G_prime node indices.

    - ancestors on any node or edge are also selected
    """
    logger.info("Extracting solution from x and y vars...")

    x_vars = vars["x"]
    y_vars = vars["y"]

    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        logger.error(f"ERROR: Non optimal result status of {status}!")
        raise ValueError("Non optimal result status!")

    # x_var extraction
    selected_node_indices = [i for i in G_prime.node_indices() if round(model.get_value(x_vars[i])) == 1]
    solution_nodes = [G_prime[i] for i in selected_node_indices]
    solution_nodes += [G[a] for node in solution_nodes for a in node["ancestors"]]

    # y_var extraction
    selected_edges = [
        e for e in G_prime.edge_indices() if e in y_vars and round(model.get_value(y_vars[e])) == 1
    ]
    for edge in selected_edges:
        u, v = G_prime.get_edge_endpoints_by_index(edge)
        if u in selected_node_indices and v in selected_node_indices:
            solution_nodes += [G[a] for a in G_prime.get_edge_data_by_index(edge)["ancestors"]]
        else:
            u_waypoint_key = G[u]["waypoint_key"]
            v_waypoint_key = G[v]["waypoint_key"]
            ancestor_keys = [G[a]["waypoint_key"] for a in G_prime.get_edge_data_by_index(edge)["ancestors"]]
            logger.warning(
                f"  selected edge ({u}, {v}) => ({u_waypoint_key}, {v_waypoint_key}) with ancestors {ancestor_keys} missing endpoints in selected nodes, skipping..."
            )

    # Map the nodes to their G indices to induce solution subgraph
    node_key_by_index = G.attrs["node_key_by_index"]
    solution_node_indices = [node_key_by_index.inv[n["waypoint_key"]] for n in solution_nodes]

    solution_graph = G.subgraph(solution_node_indices, preserve_attrs=False)
    if len(solution_graph.node_indices()) == 0:
        logger.warning("Result is an empty solution.")
        return solution_graph

    # solution_graph.reverse()

    node_key_by_index = bidict({i: solution_graph[i]["waypoint_key"] for i in solution_graph.node_indices()})
    solution_graph.attrs = {}
    solution_graph.attrs["node_key_by_index"] = node_key_by_index
    rx_api.set_graph_terminal_sets_attribute(solution_graph, G.attrs["terminals"])

    if kwargs.get("validate_solution", False):
        validate_solution(solution_graph)

    if kwargs.get("cleanup_solution", False):
        cleanup_solution(solution_graph)

    print(f"Solution: {[n['waypoint_key'] for n in solution_graph.nodes()]}")
    print(f"Solution Cost: {sum(n['need_exploration_point'] for n in solution_graph.nodes())}")
    return solution_graph
