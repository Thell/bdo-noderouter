# MARK: TESTING
from loguru import logger
import nwsf_rust

DynamicCC = nwsf_rust.DynamicCC

use_union_find = True

if __name__ == "__main__":

    def do_query(dc, u, v):
        return dc.query(u, v)

    def test_query(dc, u, v, expected_result):
        result = do_query(dc, u, v)
        if result != expected_result:
            logger.error(f"query({u}, {v}) returned {result}, expected {expected_result}")
            return False
        else:
            logger.success(f"query({u}, {v}) returned {result}, expected {expected_result}")
            return True

    def check_current_connected_components(dc, test_nodes, expected_ccs, expected_isolates):
        logger.info("--- Testing node_connected_components... ---")

        unique_ccs_set = set()
        current_isolates = []
        visited_nodes = set()

        for node_id in sorted(test_nodes):
            if node_id not in visited_nodes:
                component = sorted(dc.node_connected_component(node_id))
                if len(component) == 1:
                    current_isolates.append(component[0])
                else:
                    unique_ccs_set.add(tuple(component))
                visited_nodes.update(component)

        current_ccs = sorted([list(c) for c in unique_ccs_set])

        logger.info(f"Expected ccs: {expected_ccs}, Expected isolates: {expected_isolates}")

        if sorted([sorted(c) for c in current_ccs]) == sorted([sorted(c) for c in expected_ccs]) and sorted(
            current_isolates
        ) == sorted(expected_isolates):
            logger.success("SUCCESS: node_connected_components match expected structure.")
            return True
        else:
            logger.error(
                f"ERROR: node_connected_components mismatch: ccs: {current_ccs}, isolates: {current_isolates}"
            )
            return False

    def test_edge_operation_simple(dc, operation_name, u, v, expected_returns):
        logger.info(f"\n--- {operation_name.capitalize()} edge ({u}, {v})... ---")
        op_func = getattr(dc, operation_name)

        actual_return = op_func(u, v)

        if actual_return not in expected_returns:
            logger.error(
                f"{operation_name.capitalize()} edge ({u}, {v}) returned {actual_return}, expected one of {expected_returns}"
            )
            return False
        else:
            logger.success(
                f"{operation_name.capitalize()} edge ({u}, {v}) returned {actual_return}, expected one of {expected_returns}"
            )
            return True

    logger.info("--- DynamicCC Test Started ---")
    overall_test_success = True

    test_case_1_nodes = [0, 1, 2, 3, 4]

    logger.info("\n--- Test Case 1: Incremental Graph Building and Connectivity Check ---")

    initial_5_nodes_adj = {0: [], 1: [], 2: [], 3: [], 4: []}
    logger.info(f"Initializing DynamicCC with adj_dict: {initial_5_nodes_adj}")
    dc_incremental = DynamicCC(adj_dict=initial_5_nodes_adj, use_union_find=use_union_find)

    logger.info("\n--- Initial State (No Edges) ---")
    current_phase_success = True
    expected_matrix_initial = [[False for _ in test_case_1_nodes] for _ in test_case_1_nodes]
    for i in range(len(test_case_1_nodes)):
        expected_matrix_initial[i][i] = True

    expected_ccs_initial = []
    expected_isolates_initial = sorted([0, 1, 2, 3, 4])

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_initial[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False

    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_initial,
        expected_isolates_initial,
    )
    overall_test_success &= current_phase_success

    if current_phase_success:
        logger.success("SUCCESS: Initial state checks passed.")
    else:
        logger.error("ERROR: Initial state checks failed!")
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 1.1: Inserting (0,1) ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "insert_edge", 0, 1, (1,))

    expected_matrix_p1_e1 = [
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, True],
    ]
    expected_ccs_p1_e1 = sorted([[0, 1]])
    expected_isolates_p1_e1 = sorted([2, 3, 4])

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p1_e1[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p1_e1,
        expected_isolates_p1_e1,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 1.2: Inserting (1,2) ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "insert_edge", 1, 2, (1,))
    expected_matrix_p1_e2 = [
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, True],
    ]
    expected_ccs_p1_e2 = sorted([[0, 1, 2]])
    expected_isolates_p1_e2 = sorted([3, 4])

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p1_e2[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p1_e2,
        expected_isolates_p1_e2,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 1.3: Inserting (3,4) ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "insert_edge", 3, 4, (1,))
    expected_matrix_p1_e3 = [
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]
    expected_ccs_p1_e3 = sorted([[0, 1, 2], [3, 4]])
    expected_isolates_p1_e3 = []

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p1_e3[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p1_e3,
        expected_isolates_p1_e3,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 1.4: Inserting (2,3) to connect everything ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "insert_edge", 2, 3, (1,))
    expected_matrix_p1_e4 = [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]
    expected_ccs_p1_e4 = sorted([[0, 1, 2, 3, 4]])
    expected_isolates_p1_e4 = []

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p1_e4[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p1_e4,
        expected_isolates_p1_e4,
    )
    overall_test_success &= current_phase_success

    if current_phase_success:
        logger.success("SUCCESS: Phase 1: All edge insertions and connectivity checks passed.")
    else:
        logger.error("ERROR: Phase 1: Some edge insertions or connectivity checks failed.")
    overall_test_success &= current_phase_success

    logger.info("\n--- Test Case 2: Deleting Edges and Connectivity Check (on incremental graph) ---")

    logger.info("\n--- Phase 2.1: Deleting (2,3) to split ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "delete_edge", 2, 3, (1, 2))
    expected_matrix_p2_e1 = [
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]
    expected_ccs_p2_e1 = sorted([[0, 1, 2], [3, 4]])
    expected_isolates_p2_e1 = []

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p2_e1[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p2_e1,
        expected_isolates_p2_e1,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 2.2: Deleting (1,2) to further split ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "delete_edge", 1, 2, (1, 2))
    expected_matrix_p2_e2 = [
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, True, False, False],
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]
    expected_ccs_p2_e2 = sorted([[0, 1], [3, 4]])
    expected_isolates_p2_e2 = sorted([2])

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p2_e2[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p2_e2,
        expected_isolates_p2_e2,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 2.3: Re-inserting (1,2) to connect again ---")
    current_phase_success = test_edge_operation_simple(dc_incremental, "insert_edge", 1, 2, (1,))
    expected_matrix_p2_e3 = [
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]
    expected_ccs_p2_e3 = sorted([[0, 1, 2], [3, 4]])
    expected_isolates_p2_e3 = []

    for n1_idx in range(len(test_case_1_nodes)):
        for n2_idx in range(len(test_case_1_nodes)):
            n1 = test_case_1_nodes[n1_idx]
            n2 = test_case_1_nodes[n2_idx]
            expected_connected = expected_matrix_p2_e3[n1_idx][n2_idx]
            if not test_query(dc_incremental, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_incremental,
        test_case_1_nodes,
        expected_ccs_p2_e3,
        expected_isolates_p2_e3,
    )
    overall_test_success &= current_phase_success

    if current_phase_success:
        logger.success("SUCCESS: Test Case 2: Deleting/Re-inserting edges passed.")
    else:
        logger.error("ERROR: Test Case 2: Deleting/Re-inserting edges failed.")
    overall_test_success &= current_phase_success

    logger.info("\n--- Test Case 3: Binary Tree-like Graph ---")
    current_phase_success = True
    num_nodes_tree = 15
    tree_nodes_list = list(range(num_nodes_tree))

    initial_tree_adj = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5, 6], 3: [1], 4: [1], 5: [2], 6: [2]}
    for i in range(7, num_nodes_tree):
        initial_tree_adj[i] = []

    expected_matrix_tree_common = [[False for _ in range(num_nodes_tree)] for _ in range(num_nodes_tree)]
    expected_ccs_tree_common = sorted([[0, 1, 2, 3, 4, 5, 6]])
    expected_isolates_tree_common = sorted(list(range(7, num_nodes_tree)))

    for component in expected_ccs_tree_common:
        for i_node in component:
            for j_node in component:
                expected_matrix_tree_common[i_node][j_node] = True
    for i_node in range(num_nodes_tree):
        expected_matrix_tree_common[i_node][i_node] = True

    logger.info(f"Building a 'binary tree-like' structure of {num_nodes_tree} nodes (0 is root):")
    logger.info(f"Initializing DynamicCC with adj_dict: {initial_tree_adj}")
    cc_tree_adj_init = DynamicCC(adj_dict=initial_tree_adj, use_union_find=use_union_find)
    logger.info("Finished initializing binary tree graph (via adj_dict).")

    logger.info("\n--- Connectivity Checks for Binary Tree Graph (Initial State - via adj_dict) ---")
    for n1_idx in range(len(tree_nodes_list)):
        for n2_idx in range(len(tree_nodes_list)):
            n1 = tree_nodes_list[n1_idx]
            n2 = tree_nodes_list[n2_idx]
            expected_connected = expected_matrix_tree_common[n1_idx][n2_idx]
            if not test_query(cc_tree_adj_init, n1, n2, expected_connected):
                current_phase_success = False

    current_phase_success &= check_current_connected_components(
        cc_tree_adj_init,
        tree_nodes_list,
        expected_ccs_tree_common,
        expected_isolates_tree_common,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 3.0.5: Building Binary Tree using insert_edge ---")

    initial_empty_adj = {i: [] for i in tree_nodes_list}
    logger.info(f"Initializing new DynamicCC with adj_dict: {initial_empty_adj}")
    cc_tree_insert_edge = DynamicCC(adj_dict=initial_empty_adj, use_union_find=use_union_find)
    logger.info(
        f"Initialized new DynamicCC with all {num_nodes_tree} nodes as isolates for insert_edge test."
    )

    edges_to_insert = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

    # fmt:off
    expected_states_3_0_5 = [
        {  # After (0,1)
            "matrix": [
                [
                    True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, ], ],
                    "ccs": [[0, 1]],"isolates": sorted(list(range(2, num_nodes_tree))),
        },
        {  # After (0,2)
            "matrix": [
                [
                    True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, ], ],
                    "ccs": [[0, 1, 2]],"isolates": sorted(list(range(3, num_nodes_tree))),
        },
        {  # After (1,3)
            "matrix": [
                [
                    True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, ], ],
                    "ccs": [[0, 1, 2, 3]],"isolates": sorted(list(range(4, num_nodes_tree))),
        },
        {  # After (1,4)
            "matrix": [
                [
                    True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, ], ],
                    "ccs": [[0, 1, 2, 3, 4]],"isolates": sorted(list(range(5, num_nodes_tree))),
        },
        {  # After (2,5)
            "matrix": [
                [
                    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ], [
                    True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, ], [
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, ], ],
                    "ccs": [[0, 1, 2, 3, 4, 5]],"isolates": sorted(list(range(6, num_nodes_tree))),
        },
        {  # After (2,6) - Final state for the tree
            "matrix": expected_matrix_tree_common,
            "ccs": expected_ccs_tree_common,
            "isolates": expected_isolates_tree_common,
        },
    ]
    # fmt:on

    insert_edge_phase_success = True

    for i, (u, v) in enumerate(edges_to_insert):
        logger.info(f"--- Inserting edge ({u}, {v})... ---")

        # ONLY test the return value of insert_edge here
        if not test_edge_operation_simple(cc_tree_insert_edge, "insert_edge", u, v, (1,)):
            insert_edge_phase_success = False

        # THEN, get the expected state for *this specific step*
        expected_matrix_for_this_step = expected_states_3_0_5[i]["matrix"]
        expected_ccs_for_this_step = expected_states_3_0_5[i]["ccs"]
        expected_isolates_for_this_step = expected_states_3_0_5[i]["isolates"]

        # Now perform connectivity checks for this step's state
        logger.info(f"--- Connectivity query checks after insert_edge ({u}, {v}) ---")
        for n1_idx in range(len(tree_nodes_list)):
            for n2_idx in range(len(tree_nodes_list)):
                n1 = tree_nodes_list[n1_idx]
                n2 = tree_nodes_list[n2_idx]
                expected_connected = expected_matrix_for_this_step[n1_idx][n2_idx]  # type: ignore
                if not test_query(cc_tree_insert_edge, n1, n2, expected_connected):
                    insert_edge_phase_success = False

        # Now perform connected components check for this step's state
        if not check_current_connected_components(
            cc_tree_insert_edge, tree_nodes_list, expected_ccs_for_this_step, expected_isolates_for_this_step
        ):
            insert_edge_phase_success = False
            logger.error(f"ERROR: Component state mismatch after inserting ({u}, {v})")
        else:
            logger.success(f"SUCCESS: Component state correct after inserting ({u}, {v})")

    if insert_edge_phase_success:
        logger.success("SUCCESS: Phase 3.0.5: Binary Tree built via insert_edge passed.")
    else:
        logger.error("ERROR: Phase 3.0.5: Binary Tree built via insert_edge failed.")
    overall_test_success &= insert_edge_phase_success

    logger.info("\n--- Phase 3.1: Deleting edge (0, 1) in Binary Tree Graph ---")
    current_phase_success = test_edge_operation_simple(cc_tree_adj_init, "delete_edge", 0, 1, (1, 2))

    expected_matrix_tree_after_del = [[False for _ in range(num_nodes_tree)] for _ in range(num_nodes_tree)]

    expected_comp_1_after_del = sorted([0, 2, 5, 6])
    expected_comp_2_after_del = sorted([1, 3, 4])
    expected_isolates_3_1 = sorted(list(range(7, num_nodes_tree)))

    for i_node in expected_comp_1_after_del:
        for j_node in expected_comp_1_after_del:
            expected_matrix_tree_after_del[i_node][j_node] = True

    for i_node in expected_comp_2_after_del:
        for j_node in expected_comp_2_after_del:
            expected_matrix_tree_after_del[i_node][j_node] = True

    for i_node in range(num_nodes_tree):
        expected_matrix_tree_after_del[i_node][i_node] = True

    for n1_idx in range(len(tree_nodes_list)):
        for n2_idx in range(len(tree_nodes_list)):
            n1 = tree_nodes_list[n1_idx]
            n2 = tree_nodes_list[n2_idx]
            expected_connected = expected_matrix_tree_after_del[n1_idx][n2_idx]
            if not test_query(cc_tree_adj_init, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        cc_tree_adj_init,
        tree_nodes_list,
        [expected_comp_1_after_del, expected_comp_2_after_del],
        expected_isolates_3_1,
    )
    overall_test_success &= current_phase_success

    if use_union_find:
        logger.info("\n--- Phase 3.1.1: Explicit Reroot Check after Deleting (0,1) ---")
        current_phase_success_reroot = True

        root_of_134 = cc_tree_adj_init.get_f(1)

        if cc_tree_adj_init.get_f(3) == root_of_134 and cc_tree_adj_init.get_f(4) == root_of_134:
            logger.success(f"SUCCESS: Nodes 1, 3, 4 are in the same component (root: {root_of_134}).")
        else:
            logger.error("ERROR: Nodes 1, 3, 4 are not in the same component after (0,1) deletion.")
            current_phase_success_reroot = False

        root_of_0256 = cc_tree_adj_init.get_f(0)

        if (
            cc_tree_adj_init.get_f(2) == root_of_0256
            and cc_tree_adj_init.get_f(5) == root_of_0256
            and cc_tree_adj_init.get_f(6) == root_of_0256
        ):
            logger.success(f"SUCCESS: Nodes 0, 2, 5, 6 are in the same component (root: {root_of_0256}).")
        else:
            logger.error("ERROR: Nodes 0, 2, 5, 6 are not in the same component after (0,1) deletion.")
            current_phase_success_reroot = False

        if root_of_134 == root_of_0256:
            logger.error("ERROR: The two new components are not distinct (their roots are the same).")
            current_phase_success_reroot = False
        else:
            logger.success("SUCCESS: The two new components have distinct roots.")

        overall_test_success &= current_phase_success_reroot

    logger.info("\n--- Phase 3.2: Inserting edge (1, 2) in Binary Tree Graph ---")
    current_phase_success = test_edge_operation_simple(cc_tree_adj_init, "insert_edge", 1, 2, (1,))

    for n1_idx in range(len(tree_nodes_list)):
        for n2_idx in range(len(tree_nodes_list)):
            n1 = tree_nodes_list[n1_idx]
            n2 = tree_nodes_list[n2_idx]
            expected_connected = expected_matrix_tree_common[n1_idx][n2_idx]
            if not test_query(cc_tree_adj_init, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        cc_tree_adj_init,
        tree_nodes_list,
        expected_ccs_tree_common,
        expected_isolates_tree_common,
    )
    overall_test_success &= current_phase_success

    if current_phase_success:
        logger.success("SUCCESS: Test Case 3: Binary Tree-like Graph Test passed.")
    else:
        logger.error("ERROR: Test Case 3: Binary Tree-like Graph Test failed.")
    overall_test_success &= current_phase_success

    logger.info("\n--- Test Case 4: Isolated Nodes Edge Operations ---")
    test_case_4_nodes = [0, 1, 2, 3]
    initial_4_isolates_adj = {i: [] for i in test_case_4_nodes}
    logger.info(f"Initializing DynamicCC with adj_dict: {initial_4_isolates_adj}")
    dc_isolates = DynamicCC(adj_dict=initial_4_isolates_adj, use_union_find=use_union_find)

    logger.info("\n--- Phase 4.1: Inserting edge (0,1) between isolates ---")
    current_phase_success = test_edge_operation_simple(dc_isolates, "insert_edge", 0, 1, (1,))
    expected_matrix_4_1 = [[False for _ in test_case_4_nodes] for _ in test_case_4_nodes]
    expected_ccs_4_1 = sorted([[0, 1]])
    expected_isolates_4_1 = sorted([2, 3])
    for i in range(len(test_case_4_nodes)):
        expected_matrix_4_1[i][i] = True
    expected_matrix_4_1[0][1] = expected_matrix_4_1[1][0] = True

    for n1_idx in range(len(test_case_4_nodes)):
        for n2_idx in range(len(test_case_4_nodes)):
            n1 = test_case_4_nodes[n1_idx]
            n2 = test_case_4_nodes[n2_idx]
            expected_connected = expected_matrix_4_1[n1_idx][n2_idx]
            if not test_query(dc_isolates, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_isolates,
        test_case_4_nodes,
        expected_ccs_4_1,
        expected_isolates_4_1,
    )
    overall_test_success &= current_phase_success

    logger.info("\n--- Phase 4.2: Deleting edge (0,1) to re-isolate ---")
    current_phase_success = test_edge_operation_simple(dc_isolates, "delete_edge", 0, 1, (1, 2))
    expected_matrix_4_2 = [[False for _ in test_case_4_nodes] for _ in test_case_4_nodes]
    expected_ccs_4_2 = []
    expected_isolates_4_2 = sorted([0, 1, 2, 3])
    for i in range(len(test_case_4_nodes)):
        expected_matrix_4_2[i][i] = True

    for n1_idx in range(len(test_case_4_nodes)):
        for n2_idx in range(len(test_case_4_nodes)):
            n1 = test_case_4_nodes[n1_idx]
            n2 = test_case_4_nodes[n2_idx]
            expected_connected = expected_matrix_4_2[n1_idx][n2_idx]
            if not test_query(dc_isolates, n1, n2, expected_connected):
                current_phase_success = False
    current_phase_success &= check_current_connected_components(
        dc_isolates,
        test_case_4_nodes,
        expected_ccs_4_2,
        expected_isolates_4_2,
    )
    overall_test_success &= current_phase_success

    if overall_test_success:
        logger.success("All DynamicCC tests passed successfully!")
    else:
        logger.critical("Some DynamicCC tests FAILED!")
