# api_dimacs_nwstp.py

# The primary functionality required is to create NWSTP problems and work with
# them in DIMACS format and to read DIMACS format solutions.

import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import rustworkx as rx
from bidict import bidict
from highspy import Highs, HighsModelStatus, highs_var
from loguru import logger

import api_data_store as ds
from api_common import PAYLOAD_WEIGHT_KEY
from api_highs_solver import HiGHSVarMap, create_model, get_highs
from api_rx_pydigraph import subgraph_stable

MAGIC_HEADER = "33D32945 STP File, STP Format Version 1.0"


@dataclass(slots=True)
class TreeProblem:
    instance_id: str
    block_key: tuple[int, ...]
    terminals: Sequence[int]

    adj_map: Mapping[int, Iterable[int]]
    node_weight_map: Mapping[int, int]
    node_index_map: Mapping[int, int]

    dimacs_id_map: Mapping[int, int] | None = None
    inv_dimacs_id_map: Mapping[int, int] | None = None

    enable_super_root_index: int = 0

    do_debug: bool = False
    mip_validation: bool = False

    @property
    def complexity(self) -> int:
        return (3 ** len(self.terminals) * len(self.adj_map)) + (2 ** len(self.terminals)) * (
            len(self.adj_map) ** 2
        )


# --- Example NWSTP problem format ---
#
# 33D32945 STP File, STP Format Version 1.0
# SECTION Comment
# Name    "instance_id"
# Creator "Thell Fowler"
# Remark  "block_key"
# END
# SECTION Graph
# Nodes 154
# Edges 204
# E 1 5 0
# ...
# END
# SECTION Terminals
# Terminals 7
# T 18
# ...
# T 118
# END
# SECTION NodeWeights
# NW 0
# ...
# NW 0
# END
# EOF

# --- Example DIMACS solution format ---
#
# SECTION Comment
# Name filename.stp
# Problem RPCST
# Program SCIP-Jack
# Version 2.0
# End
# SECTION Solutions
# End
# SECTION Run
# Threads 1
# Time 0.0
# Primal     49.000000000
# End
# SECTION Finalsolution
# Vertices 35
# V 46
# --->8---
# V 198
# Edges 34
# E 46 85
# --->8---
# E 196 198
# End


class DimacsNWSTPProblem:
    """Parser for DIMACS STP (SteinLib / NWSTP) files."""

    stp_filepath: Path | None = None

    # Metadata
    name: str | None  # intance_id
    creator: str | None
    remark: str | None  # block_key

    # Graph
    num_nodes: int
    num_edges: int
    edge_list: MutableSequence[tuple[int, int]]
    graph: rx.PyDiGraph | None = None

    # Terminals
    num_terminals: int
    terminals: MutableSequence[int]

    # Node Weights (sequential)
    node_weights: MutableMapping[int, int]

    # Solution
    solution_nodes: MutableSequence[int]
    solution_edges: MutableSequence[tuple[int, int]]
    objective_value: int
    solution_graph: rx.PyDiGraph | None = None

    def __init__(self):
        # Initialize attributes
        self.stp_filepath = self.name = self.creator = self.remark = None

        self.num_nodes = 0
        self.num_edges = 0
        self.edge_list: list[tuple[int, int, int]] = []

        self.num_terminals = 0
        self.terminals: list[int] = []

        self.node_weights: dict[int, int] = {}

        self.solution_nodes: list[int] = []
        self.solution_edges: list[tuple[int, int]] = []
        self.objective_value = 0

    @property
    def stp_path(self) -> Path:
        stp_path = self.ensure_stp_filepath()
        return stp_path

    @property
    def sol_path(self) -> Path:
        stp_path = self.ensure_stp_filepath()
        return stp_path.with_suffix(stp_path.suffix + "log")

    def ensure_stp_filepath(self) -> Path:
        if self.stp_filepath is None:
            self.write_stp_file()

        assert self.stp_filepath
        return self.stp_filepath

    @classmethod
    def from_filename(cls, filename: str) -> Self:
        """Reads a DIMACS STP file into the DIMACS problem."""
        stp_filepath = Path(filename)
        if not stp_filepath.exists():
            raise FileNotFoundError(f"STP file not found: {filename}")

        with open(stp_filepath, "r", encoding="UTF-8") as f:
            lines = f.readlines()

        problem = cls()
        problem.stp_filepath = stp_filepath

        i = 0
        current_section: str | None = None

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
                        problem.name = value
                    case "CREATOR":
                        problem.creator = value
                    case "REMARK":
                        problem.remark = value

            elif current_section == "Graph":
                match keyword:
                    case "NODES":
                        problem.num_nodes = int(parts[1])
                    case "EDGES":
                        problem.num_edges = int(parts[1])
                    case "E":
                        u = int(parts[1])
                        v = int(parts[2])
                        problem.edge_list.append((u, v))

            elif current_section == "Terminals":
                match keyword:
                    case "TERMINALS":
                        problem.num_terminals = int(parts[1])
                    case "T":
                        problem.terminals.append(int(parts[1]))

            elif current_section == "NodeWeights":
                if keyword == "NW":
                    weight = int(parts[1])
                    node_id = len(problem.node_weights) + 1
                    problem.node_weights[node_id] = weight

        if problem.num_nodes == 0:
            raise ValueError("No Nodes count found in Graph section.")

        return problem

    @classmethod
    def from_treeproblem(cls, tree_problem: TreeProblem, creator="BDO NodeRouter Fuzzer") -> Self:
        """Converts a tree problem into a DIMACS STP problem.
        NOTE: TreeProblem indices are 0 based, DIMACS indices are 1 based; both are contiguous
        """
        logger.trace("      Creating DIMACS problem...")

        # This naturally disables super-root since super-root is always the last index
        # and only has incoming edges, thus u is always > v
        adj_map = tree_problem.adj_map
        dimacs_id_map = tree_problem.dimacs_id_map
        assert dimacs_id_map

        edges = []
        for u, nbrs in adj_map.items():
            for v in nbrs:
                if u < v:
                    edges.append((dimacs_id_map[u], dimacs_id_map[v]))

        enable_super_root_index = tree_problem.enable_super_root_index
        if enable_super_root_index > 0:
            logger.warning(f"        Enabling super-root node {enable_super_root_index}...")
            for nbr in adj_map[enable_super_root_index]:
                edges.append((dimacs_id_map[nbr], dimacs_id_map[enable_super_root_index]))

        # NOTE: There is a bug in scipstp requiring the edges to be sorted in some cases...
        edges = sorted(edges)

        problem = cls()

        problem.name = tree_problem.instance_id
        problem.creator = creator
        problem.remark = str(tree_problem.block_key)

        problem.num_nodes = len(tree_problem.node_weight_map)
        problem.num_edges = len(edges)

        problem.edge_list = edges

        problem.num_terminals = len(tree_problem.terminals)
        problem.terminals = [dimacs_id_map[i] for i in tree_problem.terminals]

        problem.node_weights = {dimacs_id_map[i]: w for i, w in tree_problem.node_weight_map.items()}

        return problem

    def load_mip_solution(self, model: Highs, vars: dict[str, HiGHSVarMap]):
        """Populates the DIMACS problem solution from a MIP solution."""
        x_vars = cast(Sequence[highs_var], vars["x"].values())
        solution = [i for i, v in enumerate(model.variableValues(x_vars), start=1) if round(v) == 1]
        cost = sum(self.node_weights[i] for i in solution)

        # All node indices need +1 offset from PyDiGraph to match DIMACS node ids
        graph = self.as_pydigraph()
        subG = subgraph_stable(graph, [i - 1 for i in solution])
        edges = [(u + 1, v + 1) for u, v in subG.edge_list()]

        self.solution_nodes = solution
        self.solution_edges = edges
        self.objective_value = cost
        self.solution_graph = subG

    def load_scip_solution(self):
        """Reads a DIMACS STP solution file into the DIMACS problem."""
        if self.stp_filepath is None:
            raise ValueError("DIMACS problem has no filename.")

        solution_nodes: list[int] = []
        solution_edges: list[tuple[int, int]] = []

        sol_path = self.stp_filepath.with_suffix(Path(self.stp_filepath).suffix + "log")
        with sol_path.open("r", encoding="utf-8") as f:
            section = None

            for line in f:
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == "SECTION":
                    section = parts[1] if len(parts) > 1 else None
                    continue

                # NOTE: Until we have clarification on why some solution files contain
                #       "solutions" and an objective and others don't we will just
                #       manually parse _only_ the FinalSolution section and get the cost
                #       from the problem weights.
                if section == "Finalsolution":
                    if parts[0] == "V":
                        solution_nodes.append(int(parts[1]))
                    if parts[0] == "E":
                        u = int(parts[1])
                        v = int(parts[2])
                        solution_edges.append((u, v))
                    continue

        if not solution_nodes:
            raise RuntimeError("No solution nodes found (Finalsolution missing)")

        self.solution_nodes = solution_nodes
        self.solution_edges = solution_edges
        cost = sum(self.node_weights[node_id] for node_id in solution_nodes)
        self.objective_value = cost

    def validate_solution_reachability(self) -> bool:
        """Validates reachability of the solution using rustworkx."""
        # NOTE: DIMACS indices are 1 based and rustworkx are 0 based
        graph = self.as_pydigraph()
        subG = subgraph_stable(graph, [i - 1 for i in self.solution_nodes])
        stp_terminals = self.terminals
        root = stp_terminals[0]
        terminals = stp_terminals[1:] if len(stp_terminals) > 1 else []
        for terminal in terminals:
            if not rx.has_path(subG, root - 1, terminal - 1):
                return False
        return True

    def solve_using_scipstp(self):
        """Solves the DIMACS problem using scipstp (scip-jack)."""
        stp_path = self.ensure_stp_filepath()

        sol_path = self.sol_path
        if sol_path.exists():
            sol_path.unlink()

        if not ds.is_file("scipstp.set"):
            logger.error("SCIPSTP settings file not found")
            raise FileNotFoundError
        settings_path = ds.path().joinpath("scipstp.set")

        result = subprocess.run(
            ["scipstp.exe", "-f", stp_path.name, "-s", str(settings_path.resolve())],
            cwd=str(stp_path.parent),
            capture_output=True,
            text=False,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"SCIPSTP failed on {stp_path}")
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("SCIPSTP solve failed")

        if not sol_path.exists():
            raise RuntimeError(f"SCIPSTP did not produce a solution file: {sol_path}")

        self.load_scip_solution()

    def solve_using_mip(self, **kwargs):
        """The problem is solved using MIP and the solution attributes are populated."""
        model, vars = self.as_mip_model(**kwargs)
        model.optimize()
        if model.getModelStatus() == HighsModelStatus.kInfeasible:
            raise ValueError("Infeasible!")
        self.load_mip_solution(model, vars)

    def validate_using_mip(self, **kwargs) -> int:
        """
        The scip solution is validated using the MIP model.

        Returns:
        - -1: scip solution is an invalid solution (non-reachable terminals)
        - 0: scip solutin is optimal
        - > 0: scip solution is valid but suboptimal and the MIP solution replaces the scip solution
        """
        scip_cost = self.objective_value
        scip_solution = self.solution_nodes
        if not scip_solution or not scip_cost:
            logger.error("No scip solution or cost found, nothing to validate. Skipping...")
            return -1

        if not self.validate_solution_reachability():
            return -1

        model, vars = self.as_mip_model(**kwargs)

        graph = self.as_pydigraph()
        model.addConstr(
            model.qsum(
                vars["x"][i] * graph[i][PAYLOAD_WEIGHT_KEY] for i in self.as_pydigraph().node_indices()
            )
            <= scip_cost - 1
        )
        model.optimize()

        # Validate optimality...
        # If the model is infeasible, then the scip solution is optimal
        if model.getModelStatus() == HighsModelStatus.kInfeasible:
            return 0

        # Otherwise, the solution is suboptimal
        logger.warning("SCIP solution is valid but suboptimal; replacing it with the optimal MIP solution...")
        self.load_mip_solution(model, vars)

        return self.objective_value

    def write_stp_file(self):
        if self.stp_filepath is None:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".stp"
            ) as filepath:
                self.stp_filepath = Path(filepath.name)
                filepath.close()

        path = self.stp_filepath

        with path.open("w", encoding="utf-8") as f:
            f.write(f"{MAGIC_HEADER}\n")
            f.write("SECTION Comment\n")
            f.write(f'Name    "{self.name}"\n')
            f.write(f'Creator "{self.creator}"\n')
            f.write(f'Remark  "{self.remark}"\n')
            f.write("END\n")

            f.write("SECTION Graph\n")
            f.write(f"Nodes {self.num_nodes}\n")
            f.write(f"Edges {self.num_edges}\n")
            for u, v in self.edge_list:
                f.write(f"E {u} {v} 0\n")
            f.write("END\n")

            f.write("SECTION Terminals\n")
            f.write(f"Terminals {self.num_terminals}\n")
            for t in self.terminals:
                f.write(f"T {t}\n")
            f.write("END\n")

            f.write("SECTION NodeWeights\n")
            for w in self.node_weights.values():
                f.write(f"NW {w}\n")
            f.write("END\n")

            f.write("EOF\n")

        assert self.stp_filepath.exists()

    def to_mapped_solution(self, mapping: Mapping[int, int]) -> list[int]:
        return [mapping[n] for n in self.solution_nodes]

    def as_mip_model(self, **kwargs) -> tuple[Highs, dict[str, HiGHSVarMap]]:
        graph = self.as_pydigraph()
        config = ds.get_config("dimacs_stp_mip_config")
        highs = get_highs(config)
        if kwargs.get("parallel"):
            highs.setOptionValue("parallel", kwargs["parallel"])
        model, vars = create_model(highs, graph=graph, **kwargs)

        return model, vars

    def as_pydigraph(self) -> rx.PyDiGraph:
        """
        Convert DIMACS STP to rustworkx PyDiGraph

        Requirements fulfilled:
        - Bidirectional directed graph (anti-parallel arcs for every undirected edge)
        - node payloads: {NODE_PAYLOAD_KEY: weight}
        - graph.attrs["node_key_by_index"]: bidict (index ↔ original node key)
        - graph.attrs["terminals"]: list of terminal keys
        - graph.attrs["terminal_sets"]: dict mapping root to list of terminals
        """
        # NOTE: DIMACS indices are 1 based, rustworkx are 0 based, both are contiguous

        if self.graph is not None:
            return self.graph

        G_undirected = rx.PyGraph()
        node_payloads = [{PAYLOAD_WEIGHT_KEY: weight} for weight in self.node_weights.values()]
        node_indices = G_undirected.add_nodes_from(node_payloads)
        node_key_by_index = bidict(zip(node_indices, self.node_weights.keys()))

        for u, v in self.edge_list:
            # Convert DIMACS edge node ids to rustworkx indices
            G_undirected.add_edge(u - 1, v - 1, 0)

        graph: rx.PyDiGraph = G_undirected.to_directed()

        # Convert DIMACS terminal node ids to rustworkx indices
        graph.attrs = {
            "node_key_by_index": node_key_by_index,
            "terminals": [t - 1 for t in self.terminals],
            "terminal_sets": {
                self.terminals[0] - 1: [t - 1 for t in self.terminals[1:]] if len(self.terminals) > 1 else []
            },
        }

        self.solution_graph = graph
        return graph
