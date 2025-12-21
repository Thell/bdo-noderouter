# api_highs_solver.py

from dataclasses import dataclass
from multiprocessing import cpu_count
from threading import Lock, Thread
import queue
import time

from highspy import Highs, HighsModelStatus, kHighsInf, ObjSense
from loguru import logger
import numpy as np
import rustworkx as rx

from api_rx_pydigraph import subgraph_stable
from api_exploration_data import SUPER_ROOT


@dataclass
class Incumbent:
    id: int
    lock: Lock
    value: int
    solution: np.ndarray
    provided: list[bool]


def get_highs(config: dict) -> Highs:
    """Returns a configured HiGHS instance using 'config["solver"]' options."""
    highs = Highs()
    options = {k: v for k, v in config["solver"].items()}
    for option_name, option_value in options.items():
        # Non-standard HiGHS options need filtering...
        if option_name not in ["num_threads", "mip_improvement_timeout"]:
            highs.setOptionValue(option_name, option_value)
    return highs


def create_model(model: Highs, **kwargs) -> tuple[Highs, dict]:
    """Populates HiGHS model using the graph in kwargs['graph'].

    Requires:
      - the kwarg 'graph'
      - a graph with attrs member containing 'terminal_sets' and 'terminals'
      - requires nodes with a payload containing 'need_exploration_point'
    """
    logger.debug("Creating MIP Baseline model using highspy...")

    if "graph" not in kwargs:
        raise LookupError("'graph' must be in kwargs!")

    G = kwargs["graph"]
    if not isinstance(G, rx.PyDiGraph):
        raise TypeError("'graph' must be a rustworkx.PyDiGraph! Call '.to_directed' on a PyGraph if needed.")

    if "terminal_sets" not in G.attrs or "terminals" not in G.attrs:
        raise LookupError("Graph must have 'terminal_sets' and 'terminals' attributes in attrs member!")

    terminal_sets = G.attrs.get("terminal_sets", {})

    start_time = time.time()

    # Variables
    # Node selection for each node in graph to determine if node is in forest.
    x = model.addBinaries(G.node_indices())

    # Flow on edges f^k_{ij} for each k in [K] and each {i,j} (flow for root k on arc from i to j)
    f_k = {}
    for k, terminal_set in terminal_sets.items():
        f_ub = len(terminal_set)
        for i, j in G.edge_list():
            f_k[(k, i, j)] = model.addVariable(lb=0, ub=f_ub)

    # Objective: Minimize total node weight
    model.setObjective(
        model.qsum(G[i]["need_exploration_point"] * x[i] for i in G.node_indices()),
        sense=ObjSense.kMinimize,
    )

    # Constraints

    # Node/flow based constraints (terminals are sources and roots are sinks)
    all_terminals = terminal_sets.keys() | set().union(*terminal_sets.values())
    for i in G.node_indices():
        predecessors = G.predecessor_indices(i)
        successors = G.successor_indices(i)

        neighbors = set(predecessors) | set(successors)
        x_neighbors = [x[j] for j in neighbors]

        if i in all_terminals:
            # All roots and terminals must be in the solution.
            model.addConstr(x[i] == 1)

        # Neighbor selection: redundant for selection, imposes a transitive
        # property to selected nodes to improve solution runtime.
        if i in all_terminals:
            # A terminal or root must have at least one selected neighbor
            model.addConstr(model.qsum(x_neighbors) >= 1)
        else:
            # A selected intermediate must have at least two selected neighbors
            model.addConstr(model.qsum(x_neighbors) >= 2 * x[i])

        # Rooted flow based constraints:
        for k, terminal_set in terminal_sets.items():
            in_flow = model.qsum(f_k[k, j, i] for j in predecessors)
            out_flow = model.qsum(f_k[k, i, j] for j in successors)

            if i == k:
                # Flow at root node (sink)
                model.addConstr(out_flow == 0)
                model.addConstr(in_flow == len(terminal_set))

            elif i in terminal_set:
                # Flow at terminal node (source)
                model.addConstr(out_flow - in_flow == 1)

            else:
                # Node selection - any flow selects node
                model.addConstr(out_flow <= len(terminal_set) * x[i])
                # Flow conservation at intermediate node
                model.addConstr(out_flow == in_flow)

    logger.debug(f"  time to create model: {time.time() - start_time} seconds")

    vars = {"x": x}
    return model, vars


def solve(model: Highs, config: dict) -> Highs:
    solver_config = config.get("solver", {})
    mip_improvement_timeout = solver_config.get("mip_improvement_timeout", 86400)
    mip_improvement_timer = time.time()
    num_threads = solver_config.get("num_threads", max(1, cpu_count() // 2))
    result_queue = queue.Queue()

    clones = [model] + [Highs() for _ in range(num_threads - 1)]
    clones[0].HandleUserInterrupt = True
    clones[0].enableCallbacks()

    for i in range(1, num_threads):
        clones[i].passOptions(clones[0].getOptions())
        clones[i].passModel(clones[0].getModel())
        clones[i].setOptionValue("random_seed", i)
        clones[i].HandleUserInterrupt = True
        clones[i].enableCallbacks()

    obj_sense = clones[0].getObjectiveSense()[1]
    incumbent = Incumbent(
        id=0,
        lock=Lock(),
        value=2**31 if obj_sense == ObjSense.kMinimize else -(2**31),
        solution=np.zeros(clones[0].getNumCol()),
        provided=[False] * num_threads,
    )

    solution_gap = 0.03
    solution_gap_abs = 2
    thread_log_capture = [[]] * num_threads

    if obj_sense == ObjSense.kMinimize:

        def is_better(a, b):
            return a < b

    else:

        def is_better(a, b):
            return a > b

    def cbLoggingHandler(e):
        """Follow and emit the output log of the incumbent..."""
        nonlocal incumbent, thread_log_capture, clones
        thread_id = int(e.user_data)
        if thread_log_capture[thread_id] or e.message.startswith("\nSolving report"):
            with incumbent.lock:
                thread_log_capture[thread_id].append(e.message)
                for clone_id in range(num_threads):
                    if clone_id != thread_id:
                        clones[clone_id].silent()
        elif thread_id == incumbent.id:
            print(e.message, end="")

    def cbMIPImprovedSolutionHandler(e):
        """Update incumbent to best solution found so far..."""
        # Solution gap and gap abs checks limit the locks and incumbent sharing.
        nonlocal incumbent, clones, mip_improvement_timer
        value = e.data_out.objective_function_value
        value = int(value) if value != kHighsInf else value
        if is_better(value, incumbent.value) and (
            abs(value - incumbent.value) / incumbent.value >= solution_gap
            or abs(value - incumbent.value) >= solution_gap_abs
        ):
            thread_id = int(e.user_data)
            with incumbent.lock:
                mip_improvement_timer = time.time()
                incumbent.value = value
                incumbent.solution[:] = e.data_out.mip_solution
                incumbent.provided = [False] * num_threads
                incumbent.provided[thread_id] = True
                incumbent.id = thread_id
                logger.debug(f"Incumbent supplanted by thread {thread_id} with {value}")
                return

    def cbMIPUserSolutionHandler(e):
        nonlocal incumbent
        if incumbent.value == 2**31:
            return
        value = e.data_out.objective_function_value
        value = int(value) if value != kHighsInf else value
        thread_id = int(e.user_data)
        if (
            incumbent.provided[thread_id] is False
            and is_better(incumbent.value, value)
            and (
                abs(value - incumbent.value) / incumbent.value >= solution_gap
                or abs(value - incumbent.value) >= solution_gap_abs
            )
        ):
            with incumbent.lock:
                e.data_in.user_has_solution = True
                e.data_in.user_solution[:] = incumbent.solution
                incumbent.provided[thread_id] = True
                logger.debug(f"Provided incumbent to thread {thread_id} with {value}")
                return

    def cbMIPInterruptHandler(e):
        if time.time() - mip_improvement_timer >= mip_improvement_timeout:
            e.interrupt()

    for i in range(num_threads):
        clones[i].cbMipImprovingSolution.subscribe(cbMIPImprovedSolutionHandler, i)
        clones[i].cbMipUserSolution.subscribe(cbMIPUserSolutionHandler, i)
        clones[i].cbLogging.subscribe(cbLoggingHandler, i)
        clones[i].cbMipInterrupt.subscribe(cbMIPInterruptHandler, i)

    def task(clone: Highs, i: int):
        clone.solve()
        result_queue.put(i)

    for i in range(num_threads):
        Thread(target=task, args=(clones[i], i), daemon=True).start()
        time.sleep(0.1)

    first_to_finish = None
    while first_to_finish is None:
        try:
            first_to_finish = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

    for i in range(num_threads):
        clones[i].cancelSolve()

    for message in thread_log_capture[first_to_finish]:
        print(message, end="")

    return clones[first_to_finish]


def extract_solution(model: Highs, vars: dict, G: rx.PyDiGraph, config: dict) -> rx.PyDiGraph:
    """Create a subgraph from the graph consisting of the node x vars from the solved model."""
    logger.debug("Extracting solution from x vars from highspy...")

    x_vars = vars["x"]

    solution_nodes = []
    status = model.getModelStatus()
    if status == HighsModelStatus.kOptimal or HighsModelStatus.kInterrupt:
        col_values = model.getSolution().col_value
        solution_nodes = [i for i in G.node_indices() if round(col_values[x_vars[i].index]) == 1]
    else:
        logger.error(f"ERROR: Non optimal result status of {status}!")
        raise ValueError("Non optimal result status!")

    solution_graph = subgraph_stable(G, solution_nodes)
    if solution_graph.num_nodes() == 0:
        # NOTE: This is an error but not fatal because input can be contrived to cause this.
        logger.error("Result is an empty solution.")
        return solution_graph

    _cleanup_solution(solution_graph)

    return solution_graph


def _cleanup_solution(solution_graph: rx.PyDiGraph):
    """Cleanup solution: remove nodes not used in any path of terminal_sets."""
    # NOTE: Since the mip model's objective is to minimize cost and not terminal or
    # edge counts zero-cost nodes/edges can be present in the solution.
    # Following a least inclusive solution mind-set we remove these nodes.
    logger.info("Cleaning solution...")

    terminal_sets = solution_graph.attrs["terminal_sets"]
    node_key_by_index = solution_graph.attrs["node_key_by_index"]
    root_indices_in_graph = {v for v in solution_graph.node_indices() if solution_graph[v]["is_base_town"]}

    # Isolated node removal
    isolates = list(rx.isolates(solution_graph))
    isolate_cost = sum([solution_graph[i]["need_exploration_point"] for i in isolates])
    # Sanity check
    if isolate_cost > 0:
        logger.error(f"  isolates cost: {isolate_cost}")
        raise ValueError("Something is wrong with the mip model!")
    solution_graph.remove_nodes_from(isolates)

    # Unused node removal...
    # NOTE: Accumulate all nodes from root to terminal paths within solution_graph
    #       using a least inclusive methodology meaning super terminals must be connected
    #       after all other terminals and prefer connecting to an already used root.
    used_nodes = set()
    used_roots = set()

    # Normal terminal paths...
    for r_index, terminal_set in terminal_sets.items():
        r_key = node_key_by_index[r_index]
        if r_key == SUPER_ROOT:
            continue
        for t_index in terminal_set:
            if t_index in root_indices_in_graph:
                continue
            if not rx.has_path(solution_graph, r_index, t_index):
                t_key = node_key_by_index[t_index]
                raise ValueError(f"Terminal {t_key} not connected to root {r_key}!")
            path = rx.dijkstra_shortest_paths(solution_graph, r_index, t_index)
            used_nodes.update(path[t_index])
            used_roots.add(r_index)

    # Super terminal paths
    for r_index, terminal_set in terminal_sets.items():
        r_key = node_key_by_index[r_index]
        if r_key != SUPER_ROOT:
            continue
        for t_index in terminal_set:
            for potential_root in used_roots:
                if not rx.has_path(solution_graph, potential_root, t_index):
                    continue
                path = rx.dijkstra_shortest_paths(solution_graph, potential_root, t_index)
                used_nodes.update(path[t_index])
                used_nodes.add(potential_root)
                break
            else:
                for potential_root in root_indices_in_graph - used_roots:
                    if potential_root != SUPER_ROOT:
                        if not rx.has_path(solution_graph, potential_root, t_index):
                            continue
                        path = rx.dijkstra_shortest_paths(solution_graph, potential_root, t_index)
                        used_nodes.update(path[t_index])
                        used_nodes.add(potential_root)
                        break
                else:
                    t_key = node_key_by_index[t_index]
                    raise ValueError(f"Super Terminal {t_key} not connected to any base town!")

    unused_nodes = list(set(solution_graph.node_indices()) - used_nodes)
    unused_nodes_with_cost = [i for i in unused_nodes if solution_graph[i]["need_exploration_point"] > 0]
    unused_cost = 0
    for node in unused_nodes_with_cost:
        cost = solution_graph[node]["need_exploration_point"]
        unused_cost += cost
        logger.warning(f"  removing node {node_key_by_index[node]} with cost {cost}")
    if unused_cost > 0:
        logger.warning(f"  total unused cost: {unused_cost}")
        logger.warning("Check (tighten) solver configuration! mip_feasibility_tolerance should be <= 1e-5")

    solution_graph.remove_nodes_from(unused_nodes)
