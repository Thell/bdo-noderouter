use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

use crate::node_router::{ExplorationGraphData, ExplorationNodeData, NodeRouter};

#[pyclass(name = "NodeRouter", unsendable)]
pub struct PyNodeRouter {
    inner: NodeRouter,
}

#[pymethods]
impl PyNodeRouter {
    #[new]
    pub fn new(exploration_json: &str) -> PyResult<Self> {
        let str_map: BTreeMap<String, ExplorationNodeData> = serde_json::from_str(exploration_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let exploration_data: ExplorationGraphData = str_map
            .into_iter()
            .map(|(k, v)| {
                k.parse::<usize>()
                    .map(|parsed| (parsed, v))
                    .map_err(|e| PyValueError::new_err(format!("Invalid key '{}': {}", k, e)))
            })
            .collect::<PyResult<_>>()?;

        Ok(PyNodeRouter {
            inner: NodeRouter::new(&exploration_data),
        })
    }

    #[pyo3(name = "solve_for_terminal_pairs")]
    pub fn solve_for_terminal_pairs(
        &mut self,
        terminal_pairs: Vec<(usize, usize)>,
    ) -> (Vec<usize>, usize) {
        self.inner.solve_for_terminal_pairs(terminal_pairs)
    }

    pub fn set_option(&mut self, option: String, value: String) -> PyResult<()> {
        self.inner
            .set_option(&option, &value)
            .map_err(PyValueError::new_err)
    }
}
