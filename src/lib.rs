#![feature(coroutines, coroutine_trait)]

mod bridge_generator;
mod helpers_common;
mod idtree;
mod node_router;
mod weighted_combo_generator;

// Core exports, available for all builds
pub use crate::idtree::IDTree;
pub use crate::node_router::NodeRouter;

// PyO3 binding
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn nwsf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IDTree>()?;
    m.add_class::<NodeRouter>()?;
    Ok(())
}

// WASM binding
#[cfg(feature = "wasm")]
mod wasm_node_router;

#[cfg(feature = "wasm")]
pub use wasm_node_router::WasmNodeRouter;
