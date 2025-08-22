extern crate noderouter;

use std::hint::black_box;
use std::path::PathBuf;

use noderouter::node_router::ExplorationGraphData;
use noderouter::node_router::NodeRouter;

/*
Sample data: root pairs needs to be extracted from the workerman json where each entry in the
"userWorkers" contains the `tnk`` and `job["pzk"]` attributes where `tnk` is the root and
`job["pzk"]` is the terminal.
 */

fn main() {
    let exploration_data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("noderouter")
        .join("data")
        .join("clean_exploration.json");

    let exploration_data = std::fs::read_to_string(&exploration_data_path).unwrap();
    let exploration_data: serde_json::Value = serde_json::from_str(&exploration_data).unwrap();
    let exploration_data: ExplorationGraphData = serde_json::from_value(exploration_data).unwrap();

    let mut nr = NodeRouter::new(&exploration_data);

    let test_data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("noderouter")
        .join("data")
        .join("workerman")
        .join("550_250609_2037.json");

    let test_data = std::fs::read_to_string(&test_data_path).unwrap();
    let test_data: serde_json::Value = serde_json::from_str(&test_data).unwrap();

    // Extract the terminal, root pairs used in the test_data
    let terminal_pairs: Vec<(usize, usize)> = test_data
        .get("userWorkers")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|worker| {
            (
                worker["job"]["pzk"].as_u64().unwrap() as usize,
                worker["tnk"].as_u64().unwrap() as usize,
            )
        })
        .collect();

    // Repeatedly test using a blackboxed solver
    for _ in 0..500 {
        let _ = black_box(nr.solve_for_terminal_pairs(terminal_pairs.clone()));
    }
}
