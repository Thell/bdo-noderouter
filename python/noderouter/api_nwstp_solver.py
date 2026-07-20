import shutil
import sys
from pathlib import Path

import nwst_dw
from loguru import logger

from api_common import set_logger
from api_nwstp_problem import DimacsNWSTPProblem, TreeProblem

type TreeSolveResult = tuple[tuple[int, ...], int, int]

DW_MAX_TREE_COMPLEXITY = 10_000_000


def solution_mask(solution_nodes, node_index):
    mask = 0
    for u in solution_nodes:
        mask |= 1 << node_index[u]
    return mask


def solve_tree(problem: TreeProblem, solver: str = "choose") -> TreeSolveResult:
    if solver == "choose":
        solver = "dw" if problem.complexity < DW_MAX_TREE_COMPLEXITY else "scipstp"

    if problem.enable_super_root_index:
        solver = "scipstp"

    logger.trace(f"  Using solver: {solver} on problem with complexity {problem.complexity}...")

    match solver:
        case "dw":
            return solve_tree_using_dw(problem)
        case "scipstp":
            return solve_tree_using_scipstp(problem)
        case "mip":
            return solve_tree_using_mip(problem)
        case _:
            raise ValueError(f"Unknown solver: {solver}")


def solve_tree_using_dw(problem: TreeProblem) -> TreeSolveResult:
    logger.trace("solve_tree_using_dw...")

    # solve_nwst requires neighbors to be a set (for now)
    adj_map = {k: set(v) for k, v in problem.adj_map.items()}

    try:
        cost, _, solution_nodes = nwst_dw.solve_nwst(
            adj_map,
            problem.node_weight_map,
            problem.terminals,
            problem.terminals[0],
        )
    except Exception:
        logger.error(f"  Unknown DW error processing {problem.instance_id} block {problem.block_key}...")
        raise

    mask = solution_mask(solution_nodes, problem.node_index_map)

    return problem.block_key, cost, mask


def solve_tree_using_scipstp(problem: TreeProblem) -> TreeSolveResult:
    logger.trace("solve_tree_using_scipstp...")

    if not problem.dimacs_id_map or not problem.inv_dimacs_id_map:
        raise ValueError("Invalid scipstp problem: DIMACS ID mapping not found!")

    try:
        dimacs_problem = DimacsNWSTPProblem.from_treeproblem(problem)
    except Exception as err:
        logger.error(
            f"  Unknown Dimacs handling error processing {problem.instance_id} block {problem.block_key}..."
        )
        print(err, file=sys.stderr)
        raise

    err_filepath = None

    try:
        dimacs_problem.solve_using_scipstp()

        if problem.mip_validation:
            scip_cost = dimacs_problem.objective_value
            mip_result = dimacs_problem.validate_using_mip()

            if mip_result != 0:
                if mip_result == -1:
                    err_filepath = f"scipstp_errors/{problem.instance_id}_{problem.block_key}.stp"
                    err_msg = "violated demand"
                else:
                    err_filepath = f"scipstp_errors/{problem.instance_id}_{problem.block_key}_mip_cost_{mip_result}_scip_cost_{scip_cost}.stp"
                    err_msg = "suboptimal"
                raise RuntimeError("Invalid scipstp solution")

            elif logger._core.min_level >= 10:  # 10 = DEBUG  # ty:ignore[unresolved-attribute]
                logger.success(
                    f"  scipstp solution is optimal for instance {problem.instance_id} block {problem.block_key}..."
                )

    except Exception as err:
        if err_filepath is None:
            err_msg = "unknown state error"
            err_filepath = f"scipstp_errors/{problem.instance_id}_{problem.block_key}.stp"

        logger.error(f"  Invalid scipstp solution, {err_msg}!  Saving file to: {err_filepath}")
        shutil.copyfile(dimacs_problem.stp_path, err_filepath)
        print(err, file=sys.stderr)
        raise

    finally:
        for p in (dimacs_problem.stp_path, dimacs_problem.sol_path):
            try:
                Path(p).unlink()
            except FileNotFoundError:
                pass

    cost = dimacs_problem.objective_value

    solution_nodes = [problem.inv_dimacs_id_map[u] for u in dimacs_problem.solution_nodes]
    mask = solution_mask(solution_nodes, problem.node_index_map)

    return problem.block_key, cost, mask


def solve_tree_using_mip(problem: TreeProblem) -> TreeSolveResult:
    logger.trace("solve_tree_using_scipstp...}")

    if not problem.dimacs_id_map or not problem.inv_dimacs_id_map:
        raise ValueError("Invalid problem: DIMACS ID mapping not found!")

    try:
        dimacs_problem = DimacsNWSTPProblem.from_treeproblem(problem)
    except Exception as err:
        logger.error(
            f"  Unknown Dimacs handling error processing {problem.instance_id} block {problem.block_key}..."
        )
        print(err, file=sys.stderr)
        raise

    try:
        dimacs_problem.solve_using_mip()

    except Exception as err:
        err_msg = "unknown state error"
        err_filepath = f"scipstp_errors/{problem.instance_id}_{problem.block_key}.stp"

        logger.error(f"  Invalid scipstp solution, {err_msg}!  Saving file to: {err_filepath}")
        shutil.copyfile(dimacs_problem.stp_path, err_filepath)
        print(err, file=sys.stderr)
        raise

    finally:
        for p in (dimacs_problem.stp_path, dimacs_problem.sol_path):
            try:
                Path(p).unlink()
            except FileNotFoundError:
                pass

    cost = dimacs_problem.objective_value

    solution_nodes = dimacs_problem.to_mapped_solution(problem.dimacs_id_map)
    mask = solution_mask(solution_nodes, problem.node_index_map)

    return problem.block_key, cost, mask


if __name__ == "__main__":
    import argparse

    import api_data_store as ds
    from api_common import set_logger

    set_logger({"logger": {"level": "TRACE", "format": "<level>{message}</level>"}})

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="DIMACS STP file")
    args = parser.parse_args()
    filename = args.filename

    # filename = r"C:\Users\thell\Workspaces\bdo\bdo-noderouter\845487e_(152, 388, 612, 655)_mip_cost_99999_scip_cost_99999.stp"

    problem = DimacsNWSTPProblem.from_filename(filename)

    lp_filename = f"{filename}.lp"
    problem.solve_using_mip(save_lp=lp_filename)

    print(f"solution: {problem.solution_nodes}")
    print(f"cost: {problem.objective_value}")
