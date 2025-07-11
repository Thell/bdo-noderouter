# highs_model_api.py

from copy import deepcopy

import highspy
from loguru import logger
from pyoptinterface.highs import Model
import pyoptinterface as poi
import rustworkx as rx

SUPER_ROOT = 99999


def get_model(config: dict) -> Model:
    """Acquires a high Model instance configured using arguments in kwargs that match
    HiGHS model options. ()
    """
    model = Model()
    highs_options = config.get("solver", {}).get("highs", {})
    for sub_key, sub_value in highs_options.items():
        model.set_raw_parameter(sub_key, sub_value)
    # if config.get("solver", False).get("use_concurrent", False):
    model.ConcurrentSolve = True

    return model


def populate_model(model: Model, type: str, **kwargs) -> tuple[Model | highspy.Highs, dict]:
    """Populates the Model variables and constraints using a model of type and
    arguments in kwargs which is passed to the create_model function.

    Requires:
      - the kwarg 'graph'
      - a graph with attrs member containing 'terminal_sets' and 'terminals'
      - all models require nodes with a payload contained 'need_exploration_point'
      - models with node and edge weights also require 'cost' attribute on edges

    Available models types are:
    - 'ip_baseline':
        - Node Weighted Steiner Forest problem.
        - optimal solution baseline reference
        - reverse cumulative multi-commodity iteger flow
        - Requires solution extractor for binary 'x_var' variables.
    - 'ip_baseline_highspy':
        - Node Weighted Steiner Forest problem.
        - optimal solution baseline reference
        - reverse cumulative multi-commodity iteger flow
        - Requires solution extractor for binary 'x_var' variables.
    - 'nw_ew_continuous_flow':
        - Node and Edge Weighted Steiner Forest problem.
        - Continuous variable for flows
        - Binary selection variables for nodes and edges
        - Requires solution extractor for binary 'x_var' and 'y_var' variables.
        - Extractor must be able to handle ancestor extraction on nodes and edges.
    """
    if "graph" not in kwargs:
        raise LookupError("'graph' must be in kwargs!")

    graph = kwargs["graph"]
    if not isinstance(graph, rx.PyDiGraph):
        raise TypeError("'graph' must be a rustworkx.PyDiGraph! Call '.to_directed' on a PyGraph if needed.")

    if "terminal_sets" not in graph.attrs or "terminals" not in graph.attrs:
        raise LookupError("Graph must have 'terminal_sets' and 'terminals' attributes in attrs member!")

    match type.lower():
        case "ip_baseline":
            return create_ip_baseline_model(model, **kwargs)
        case "ip_baseline_highspy":
            return create_ip_baseline_model_highspy(**kwargs)
        case "nw_ew_continuous_flow":
            return create_nw_ew_continuous_flow_model(model, **kwargs)
        case _:
            raise ValueError(f"Unknown model type: {type}")


def solve_model(model: Model, graph: rx.PyDiGraph, config: dict, vars: dict | None = None) -> None:
    """Solves the given model in-place.

    - requires a graph with attrs containing 'terminal_sets' and 'terminals'.
    """
    num_nodes = len(graph.node_indices())
    num_edges = len(graph.edge_list())
    num_components = len(rx.strongly_connected_components(graph))

    if "terminal_sets" not in graph.attrs or "terminals" not in graph.attrs:
        raise LookupError("Graph must have 'terminal_sets' and 'terminals' attributes in attrs member!")

    terminal_sets = graph.attrs.get("terminal_sets", {})
    num_roots = len(terminal_sets)
    num_terminals = len(graph.attrs.get("terminals", {}))
    num_super_terminals = sum([1 for r in graph.attrs.get("terminals").values() if r == 99999])

    num_jobs = config.get("solver", {}).get("jobs", 1)

    print(
        f"\nSolving graph with: {num_nodes} nodes, {num_edges} edges, {num_roots}"
        f" roots, {num_terminals} terminals with {num_super_terminals} super terminals",
        f" and {num_components} components using  {num_jobs} threads.\n",
    )
    if all(x == 0 for x in [num_roots, num_terminals, num_super_terminals]):
        raise Warning("terminal_sets is empty! Trying to continue, expect failure...")

    if num_jobs > 1:
        from solve_par import solve_par

        assert vars is not None
        model = solve_par(model, vars, config)
    else:
        model.optimize()


def create_ip_baseline_model(model: Model, **kwargs) -> tuple[Model, dict]:
    """Node Weighted Steiner Forest problem.

    - optimal solution baseline reference
    - reverse cumulative multi-commodity integer flow

    Requires solution extractor for binary 'x_var' variables.
    """
    import time

    start_time = time.time()
    logger.info("Creating IP Baseline model...")

    G = kwargs.get("graph", None)

    if not isinstance(G, rx.PyDiGraph):
        raise TypeError("'graph' must be a rustworkx.PyDiGraph! Call '.to_directed' on a PyGraph if needed.")

    terminal_sets = G.attrs.get("terminal_sets", {})
    if not terminal_sets:
        raise LookupError("Graph must have 'terminal_sets' attribute in attrs member!")

    # Variables
    # Node selection for each node in graph to determine if node is in forest.
    x = {}
    # Edge selection y^k_{ij} for each k in [K] and each {i,j}
    y_k = {}
    # Flow on edges f^k_{ij} for each k in [K] and each {i,j} (flow for root k on arc from i to j)
    f_k = {}

    for i in G.node_indices():
        x[(i)] = model.add_variable(lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"x_{i}")

    for k, terminal_set in terminal_sets.items():
        f_ub = len(terminal_set)
        for i, j in G.edge_list():
            y_k[(k, i, j)] = model.add_variable(
                lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"y_{k}_{i}_{j}"
            )
            f_k[(k, i, j)] = model.add_variable(
                lb=0, ub=f_ub, domain=poi.VariableDomain.Integer, name=f"f_{k}_{i}_{j}"
            )

    # Objective: Minimize total node cost
    model.set_objective(
        poi.quicksum(G[i]["need_exploration_point"] * x[i] for i in G.node_indices()),
        sense=poi.ObjectiveSense.Minimize,
    )

    # Constraints

    # Node/flow based constraints
    for i in G.node_indices():
        predecessors = G.predecessor_indices(i)
        successors = G.successor_indices(i)

        for k, terminal_set in terminal_sets.items():
            in_flow = poi.quicksum(f_k[k, j, i] for j in predecessors)
            out_flow = poi.quicksum(f_k[k, i, j] for j in successors)

            # Node selection
            model.add_linear_constraint(in_flow <= len(terminal_set) * x[i])
            model.add_linear_constraint(out_flow <= len(terminal_set) * x[i])

            # Edge selection
            for j in predecessors:
                model.add_linear_constraint(f_k[(k, j, i)] <= len(terminal_set) * y_k[(k, j, i)])
            for j in successors:
                model.add_linear_constraint(f_k[(k, i, j)] <= len(terminal_set) * y_k[(k, i, j)])

            # Flow
            if i == k:
                # Flow at root
                model.add_linear_constraint(out_flow == 0)
                model.add_linear_constraint(in_flow == len(terminal_set))

            elif i in terminal_set:
                # Flow at terminal
                model.add_linear_constraint(out_flow - in_flow == 1)
                # Maximum of a single outgoing arc
                model.add_linear_constraint(poi.quicksum(y_k[(k, i, j)] for j in successors) <= 1)

            else:
                # Flow at intermediate node
                model.add_linear_constraint(out_flow - in_flow == 0)
                # Maximum of a single outgoing arc
                model.add_linear_constraint(poi.quicksum(y_k[(k, i, j)] for j in successors) <= 1)

            # No back and forth flow to same neighbor for root k
            for j in predecessors:
                if (k, j, i) in y_k and (k, i, j) in y_k:
                    model.add_linear_constraint(y_k[(k, i, j)] + y_k[(k, j, i)] <= 1)

    end_time = time.time()
    logger.warning(f"  time to  create model: {end_time - start_time} seconds")

    vars = {"x": x, "y": y_k, "f": f_k}
    return model, vars


def create_ip_baseline_model_highspy(**kwargs) -> tuple[highspy.Highs, dict]:
    """Node Weighted Steiner Forest problem.

    - optimal solution baseline reference
    - reverse cumulative multi-commodity integer flow

    Returns HiGHS model and variable dictionaries.
    """
    import time

    start_time = time.time()
    logger.info("Creating IP Baseline model using highspy...")

    G = kwargs.get("graph", None)

    if not isinstance(G, rx.PyDiGraph):
        raise TypeError("'graph' must be a rustworkx.PyDiGraph! Call '.to_directed' on a PyGraph if needed.")

    terminal_sets = G.attrs.get("terminal_sets", {})
    if not terminal_sets:
        raise LookupError("Graph must have 'terminal_sets' attribute in attrs member!")

    # Initialize HiGHS model
    model = highspy.Highs()

    # Variables
    # Node selection for each node in graph to determine if node is in forest.
    x = {}
    # Edge selection y^k_{ij} for each k in [K] and each {i,j}
    y_k = {}
    # Flow on edges f^k_{ij} for each k in [K] and each {i,j} (flow for root k on arc from i to j)
    f_k = {}

    # Node selection for each node in graph to determine if node is in forest.
    for i in G.node_indices():
        x[(i)] = model.addBinary()

    for k, terminal_set in terminal_sets.items():
        f_ub = len(terminal_set)
        for i, j in G.edge_list():
            y_k[(k, i, j)] = model.addBinary()
            f_k[(k, i, j)] = model.addIntegral(lb=0, ub=f_ub)

    # Objective: Minimize total node cost
    model.setObjective(
        model.qsum(G[i]["need_exploration_point"] * x[i] for i in G.node_indices()),
        sense=highspy.ObjSense.kMinimize,
    )

    # Constraints

    # Node/flow based constraints
    for i in G.node_indices():
        predecessors = G.predecessor_indices(i)
        successors = G.successor_indices(i)

        for k, terminal_set in terminal_sets.items():
            in_flow = model.qsum(f_k[k, j, i] for j in predecessors)
            out_flow = model.qsum(f_k[k, i, j] for j in successors)

            # Node selection
            model.addConstr(in_flow <= len(terminal_set) * x[i])
            model.addConstr(out_flow <= len(terminal_set) * x[i])

            # Edge selection
            for j in predecessors:
                model.addConstr(f_k[(k, j, i)] <= len(terminal_set) * y_k[(k, j, i)])
            for j in successors:
                model.addConstr(f_k[(k, i, j)] <= len(terminal_set) * y_k[(k, i, j)])

            # Flow
            if i == k:
                # Flow at root
                model.addConstr(out_flow == 0)
                model.addConstr(in_flow == len(terminal_set))

            elif i in terminal_set:
                # Flow at terminal
                model.addConstr(out_flow - in_flow == 1)
                # Maximum of a single outgoing arc
                model.addConstr(model.qsum(y_k[(k, i, j)] for j in successors) <= 1)

            else:
                # Flow at intermediate node
                model.addConstr(out_flow - in_flow == 0)
                # Maximum of a single outgoing arc
                model.addConstr(model.qsum(y_k[(k, i, j)] for j in successors) <= 1)

            # No back and forth flow to same neighbor for root k
            for j in predecessors:
                if (k, j, i) in y_k and (k, i, j) in y_k:
                    model.addConstr(y_k[(k, i, j)] + y_k[(k, j, i)] <= 1)

    end_time = time.time()
    logger.warning(f"  time to  create model: {end_time - start_time} seconds")

    vars = {"x": x, "y": y_k, "f": f_k}
    return model, vars


def create_nw_ew_continuous_flow_model(model: Model, **kwargs) -> tuple[Model, dict]:
    """Node and Edge Weighted Steiner Forest problem.

    NOTE: This model uses a normal forward flow from root to terminals. This means that when
    the super root is present the flow must traverse from super root through base towns
    to the terminals.

    - Node and Edge Weighted Steiner Forest problem.
    - Continuous variable for flows
    - Binary selection variables for nodes and edges
    - Requires solution extractor for binary 'x_var' and 'y_var' variables.
    - Extractor must be able to handle ancestor extraction on nodes and edges.
    """
    from bidict import bidict

    logger.info("Creating NW-EW Continuous Flow model...")

    G = kwargs.get("graph", None)
    if not isinstance(G, rx.PyDiGraph):
        raise TypeError("'graph' must be a rustworkx.PyDiGraph! Call '.to_directed' on a PyGraph if needed.")

    terminal_sets = G.attrs.get("terminal_sets", {})
    if not terminal_sets:
        raise LookupError("Graph must have 'terminal_sets' attribute in attrs member!")

    if SUPER_ROOT in terminal_sets:
        # test the arc direction between SUPER_ROOT and its neighbors
        in_neighbors = G.predecessor_indices(SUPER_ROOT)
        out_neighbors = G.successor_indices(SUPER_ROOT)
        if in_neighbors:
            logger.error("SUPER_ROOT must only have outgoing arcs to terminals!")
            raise ValueError("SUPER_ROOT must only have outgoing arcs to terminals")
        if not out_neighbors:
            logger.error("SUPER_ROOT must have outgoing arcs!")
            raise ValueError("SUPER_ROOT must have outgoing arcs")

    G = deepcopy(G.copy())
    G.reverse()

    # arc (u, v) <=> (v, u) mapping with super root arcs mapped to self.
    rev_edge_map = {}
    for e in G.edge_indices():
        u, v = G.get_edge_endpoints_by_index(e)
        if G.has_edge(v, u):
            rev_e = G.edge_indices_from_endpoints(v, u)[0]
            rev_edge_map[e] = rev_e
        else:
            rev_edge_map[e] = e
    rev_edge_map = bidict(rev_edge_map)

    # Variables
    # Node selection for each node in graph to determine if node is in forest.
    x = {}
    # Edge selection for each arc in graph to determine if edge is in forest.
    y = {}
    # Edge selection y^k_{ij} for each k in [K] and each {i,j}
    y_k = {}
    # Flow on edges f^k_{ij} for each k in [K] and each {i,j} (flow for root k on arc from i to j)
    f_k = {}

    for i in G.node_indices():
        x[i] = model.add_variable(lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"x_{i}")

    for e in G.edge_indices():
        if e < rev_edge_map[e]:
            y[e] = model.add_variable(lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"y_{e}")

    for k in terminal_sets.keys():
        for e in G.edge_indices():
            y_k[(k, e)] = model.add_variable(lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"y_{k}_{e}")
            f_k[(k, e)] = model.add_variable(lb=0, domain=poi.VariableDomain.Continuous, name=f"f_{k}_{e}")

    # Objective: Minimize total cost
    edge_costs = sum(
        G.get_edge_data_by_index(e)["cost"] * y[e] for e in G.edge_indices() if e < rev_edge_map[e]
    )
    node_costs = sum(G[i]["need_exploration_point"] * x[i] for i in G.node_indices())
    model.set_objective(edge_costs + node_costs, sense=poi.ObjectiveSense.Minimize)

    # Edge Selection
    for e in y.keys():
        # Constraint: Any flow for any k in any direction on an edge forces edge selection.
        rev_e = rev_edge_map[e]
        f_sum = poi.quicksum(f_k[(k, e)] + f_k[(k, rev_e)] for k in terminal_sets.keys())
        model.add_linear_constraint(f_sum <= len(terminal_sets) * y[e])

        # Constraint: Selected edge forces endpoint selection.
        u, v = G.get_edge_endpoints_by_index(e)
        model.add_linear_constraint(y[e] <= x[u])
        model.add_linear_constraint(y[e] <= x[v])

    # Terminal selection: all terminals of each root must be selected
    for k, terminal_set in terminal_sets.items():
        model.add_linear_constraint(poi.quicksum(x[t] for t in terminal_set) == len(terminal_set))

    # Constraints
    for i in G.node_indices():
        predecessors = G.predecessor_indices(i)
        successors = G.successor_indices(i)

        in_edges = [G.edge_indices_from_endpoints(n, i)[0] for n in predecessors]
        out_edges = [G.edge_indices_from_endpoints(i, n)[0] for n in successors]

        for k, terminal_set in terminal_sets.items():
            M = len(terminal_set)
            in_flow = poi.quicksum(f_k[(k, e)] for e in in_edges)
            out_flow = poi.quicksum(f_k[(k, e)] for e in out_edges)

            # Node selection
            model.add_linear_constraint(in_flow <= M * x[i])
            model.add_linear_constraint(out_flow <= M * x[i])

            # Flow constraints
            if i == k:
                # Flow at root k
                model.add_linear_constraint(out_flow == 1)
                model.add_linear_constraint(x[i] == 1)

            elif i in terminal_set:
                # Flow at terminal t
                epsilon = 1 / len(terminal_set)
                model.add_linear_constraint(in_flow - out_flow >= epsilon)
                model.add_linear_constraint(x[i] == 1)

            else:
                # Flow at intermediate node
                model.add_linear_constraint(out_flow == in_flow)

    # Terminal flow cap - not strictly required but testing shows this aids in convergence
    for k, terminal_set in terminal_sets.items():
        total_terminal_flow = poi.quicksum(
            poi.quicksum(
                f_k[(k, e)]
                for e in [G.edge_indices_from_endpoints(n, t)[0] for n in G.predecessor_indices(t)]
            )
            - poi.quicksum(
                f_k[(k, e)] for e in [G.edge_indices_from_endpoints(t, n)[0] for n in G.successor_indices(t)]
            )
            for t in terminal_set
        )
        model.add_linear_constraint(total_terminal_flow <= 1)

    vars = {"x": x, "y": y, "y_k": y_k, "f_k": f_k}
    return model, vars
