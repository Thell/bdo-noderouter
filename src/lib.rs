#![feature(coroutines, coroutine_trait)]

mod exploration_data;
mod fast_paths;
mod generator_bridge;
mod generator_weighted_combo;
mod gssp;
mod primal_dual;

pub mod node_router;

pub use crate::node_router::NodeRouter;

// Python PyO3 binding
#[cfg(feature = "python")]
mod python_node_router;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn noderouter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_node_router::PyNodeRouter>()?;
    Ok(())
}

// WASM binding
#[cfg(feature = "wasm")]
mod wasm_node_router;

#[cfg(feature = "wasm")]
pub use wasm_node_router::WasmNodeRouter;
