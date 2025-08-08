use js_sys::Array;
use serde_wasm_bindgen::from_value;
use wasm_bindgen::prelude::*;

use std::collections::BTreeMap;

use crate::node_router::{ExplorationGraphData, ExplorationNodeData, NodeRouter};

#[wasm_bindgen]
pub struct WasmNodeRouter {
    inner: NodeRouter,
}

#[wasm_bindgen]
impl WasmNodeRouter {
    #[wasm_bindgen(constructor)]
    pub fn new(json: &JsValue) -> Result<WasmNodeRouter, JsValue> {
        let str_map: BTreeMap<String, ExplorationNodeData> = from_value(json.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

        let exploration_data: ExplorationGraphData = str_map
            .into_iter()
            .map(|(k, v)| {
                k.parse::<usize>()
                    .map(|parsed| (parsed, v))
                    .map_err(|e| JsValue::from_str(&format!("Invalid key '{}': {}", k, e)))
            })
            .collect::<Result<_, _>>()?;
        Ok(WasmNodeRouter {
            inner: NodeRouter::new(&exploration_data),
        })
    }

    #[wasm_bindgen(js_name = solveForTerminalPairs)]
    pub fn solve_for_terminal_pairs(&mut self, terminal_pairs: Array) -> Result<Array, JsValue> {
        let rust_pairs: Result<Vec<(usize, usize)>, _> = terminal_pairs
            .iter()
            .map(|pair| {
                let pair = Array::from(&pair);
                if pair.length() != 2 {
                    return Err(JsValue::from_str("Each terminal pair must have 2 items"));
                }
                let a = pair.get(0).as_f64().ok_or("First item must be a number")? as usize;
                let b = pair.get(1).as_f64().ok_or("Second item must be a number")? as usize;
                Ok((a, b))
            })
            .collect();

        let result = self.inner.solve_for_terminal_pairs(rust_pairs?);
        let js_array = Array::new();
        for val in result {
            js_array.push(&JsValue::from_f64(val as f64));
        }

        Ok(js_array)
    }
}
