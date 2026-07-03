# api_dimacs_stp.py

r"""
Example STP format:
33D32945 STP File, STP Format Version 1.0
SECTION Comment
Name    "example.stp"
Creator "Thell Fowler"
Remark  "NWSTP - partition block for composite"
END
SECTION Graph
Nodes 154
Edges 204
E 1 5 0
...
END
SECTION Terminals
Terminals 7
T 18
...
T 118
END
SECTION NodeWeights
NW 0
...
NW 0
END
EOF
"""

import argparse
import pathlib

import rustworkx as rx
from bidict import bidict
from highspy import HighsModelStatus

import api_data_store as ds
from api_common import PAYLOAD_WEIGHT_KEY
from api_highs_solver import create_model, get_highs
from api_rx_pydigraph import subgraph_stable


class DimacsSTP:
    """Parser for DIMACS STP (SteinLib / NWSTP) files."""

    # Metadata
    name: str | None
    creator: str | None
    remark: str | None
    date: str | None
    problem: str | None

    # Graph
    num_nodes: int
    num_edges: int
    edge_list: list[tuple[int, int, int]]

    # Terminals
    num_terminals: int
    terminals: list[int]

    # Node Weights (sequential)
    node_weights: dict[int, int]

    def __init__(self, filename: str):
        # Initialize attributes
        self.name = self.creator = self.remark = self.date = self.problem = None

        self.num_nodes = 0
        self.num_edges = 0
        self.edge_list: list[tuple[int, int, int]] = []

        self.num_terminals = 0
        self.terminals: list[int] = []

        self.node_weights: dict[int, int] = {}

        filepath = pathlib.Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"STP file not found: {filename}")

        with open(filepath, "r", encoding="UTF-8") as f:
            lines = f.readlines()

        i = 0
        current_section: str | None = None
        MAGIC_HEADER = "33D32945 STP File, STP Format Version 1.0"

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith("#"):
                continue

            if line.startswith("SECTION"):
                parts = line.split(maxsplit=1)
                current_section = parts[1].strip() if len(parts) > 1 else None
                continue

            if line in ("END", "EOF"):
                if line == "EOF":
                    break
                current_section = None
                continue

            if current_section is None and MAGIC_HEADER not in line:
                raise ValueError(f"Invalid STP file header in: {filename}\n{line}")

            if not current_section:
                continue

            parts = line.split()
            if not parts:
                continue

            keyword = parts[0].upper()

            if current_section == "Comment":
                value = " ".join(parts[1:]).strip('"')
                match keyword:
                    case "NAME":
                        self.name = value
                    case "CREATOR":
                        self.creator = value
                    case "REMARK":
                        self.remark = value
                    case "DATE":
                        self.date = value
                    case "PROBLEM":
                        self.problem = value

            elif current_section == "Graph":
                match keyword:
                    case "NODES":
                        self.num_nodes = int(parts[1])
                    case "EDGES":
                        self.num_edges = int(parts[1])
                    case "E":
                        u = int(parts[1])
                        v = int(parts[2])
                        w = int(parts[3]) if len(parts) > 3 else 0
                        self.edge_list.append((u, v, w))

            elif current_section == "Terminals":
                match keyword:
                    case "TERMINALS":
                        self.num_terminals = int(parts[1])
                    case "T":
                        self.terminals.append(int(parts[1]))

            elif current_section == "NodeWeights":
                if keyword == "NW":
                    weight = int(parts[1])
                    node_id = len(self.node_weights) + 1
                    self.node_weights[node_id] = weight

        if self.num_nodes == 0:
            raise ValueError("No Nodes count found in Graph section.")


def read_dimacs_stp(filename: str) -> DimacsSTP:
    """Read and parse a DIMACS STP file."""
    return DimacsSTP(filename)


def read_dimacs_stp_into_rustworkx_graph(filename: str) -> tuple[rx.PyDiGraph, DimacsSTP]:
    """
    Read and parse a DIMACS STP file and convert it into a rustworkx graph
    compatible with api_highs_solver.py::create_model.

    Requirements fulfilled:
      - Bidirectional directed graph (anti-parallel arcs for every undirected edge)
      - node payloads: {NODE_PAYLOAD_KEY: weight}
      - graph.attrs["node_key_by_index"]: bidict (index ↔ original node key)
      - graph.attrs["terminals"]: list of terminal keys
      - graph.attrs["terminal_sets"]: dict mapping root to list of terminals
    """
    # DIMACS indices are 1 based, rustworkx are 0 based; both are contiguous
    # and DIMACS weights are ordered per node id
    stp = DimacsSTP(filename)

    # Create undirected graph first
    G_undirected = rx.PyGraph()
    node_payloads = [{PAYLOAD_WEIGHT_KEY: weight} for weight in stp.node_weights.values()]
    node_indices = G_undirected.add_nodes_from(node_payloads)
    node_key_by_index = bidict(zip(node_indices, stp.node_weights.keys()))

    for u, v, _weight in stp.edge_list:
        # Convert DIMACS edge node ids to rustworkx indices
        G_undirected.add_edge(u - 1, v - 1, 0)

    graph: rx.PyDiGraph = G_undirected.to_directed()

    # Attach required metadata
    graph.attrs = {
        "node_key_by_index": node_key_by_index,
        "terminals": [t - 1 for t in stp.terminals],
        "terminal_sets": {
            stp.terminals[0] - 1: [t - 1 for t in stp.terminals[1:]] if len(stp.terminals) > 1 else []
        },
    }

    return graph, stp


def solve_dimacs_as_highs_mcf_mip(filename: str, **kwargs) -> tuple[int, list[int]]:
    """Solve a DIMACS STP file using HiGHS MCF MIP solver.

    When scip_cost and scip_nodes are provided, the scip solution is validated
    using the MIP model.
    - cost of -1: scip solution is an invalid solution (non-reachable terminals)
    - cost of 0: scip solutin is optimal
    - cost of > 0: scip solution is valid but suboptimal and the MIP solution is returned

    Requires:
      - the kwarg 'graph'
      - a graph with attrs member containing 'terminal_sets' and 'terminals'
      - requires nodes with a payload containing 'need_exploration_point'
    """
    graph, stp = read_dimacs_stp_into_rustworkx_graph(filename)

    config = ds.get_config("dimacs_stp_mip_config")
    highs = get_highs(config)
    if kwargs.get("parallel"):
        highs.setOptionValue("parallel", kwargs["parallel"])
    model, vars = create_model(highs, graph=graph, **kwargs)

    # Are we _validating_ the scipstp solution using the MIP model?
    if (scip_cost := kwargs.get("scip_cost")) and (scip_solution := kwargs.get("scip_solution")):
        # Validate reachability...
        # NOTE: DIMACS indices are 1 based and rustworkx are 0 based
        subG = subgraph_stable(graph, [i - 1 for i in scip_solution])
        stp_terminals = stp.terminals
        root = stp_terminals[0]
        terminals = stp_terminals[1:] if len(stp_terminals) > 1 else []
        for terminal in terminals:
            if not rx.has_path(subG, root - 1, terminal - 1):
                return -1, []

        # Validate optimality...
        model.addConstr(
            model.qsum(vars["x"][i] * graph[i][PAYLOAD_WEIGHT_KEY] for i in graph.node_indices())
            <= scip_cost - 1
        )
        model.optimize()
        if model.getModelStatus() == HighsModelStatus.kInfeasible:
            return 0, []
    else:
        model.optimize()
        if model.getModelStatus() == HighsModelStatus.kInfeasible:
            raise ValueError("Infeasible!")

    # All node indices need +1 offset from PyDiGraph to match DIMACS node ids
    solution = [i for i, v in enumerate(model.variableValues(vars["x"].values()), start=1) if round(v) == 1]
    cost = sum(stp.node_weights[i] for i in solution)

    return cost, solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="DIMACS STP file")
    args = parser.parse_args()

    cost, solution = solve_dimacs_as_highs_mcf_mip(args.filename)

    # filename = (
    #     r"C:\Users\thell\Workspaces\bdo\bdo-noderouter\scipstp_errors\(0, 145)_mip_cost_37_scip_cost_38.0.stp"
    # )
    # cost, solution = solve_dimacs_as_highs_mcf_mip(filename)

    print(f"solution: {solution}")
    print(f"cost: {cost}")
