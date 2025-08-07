use wasm_bindgen::prelude::*;

use crate::node_router::NodeRouter;

#[wasm_bindgen]
pub struct WasmNodeRouter {
    inner: NodeRouter,
}

#[wasm_bindgen]
impl WasmNodeRouter {
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> WasmNodeRouter {
        WasmNodeRouter {
            inner: NodeRouter::wasm_new(json),
        }
    }

    #[wasm_bindgen]
    pub fn solve_for_terminal_pairs(&mut self, pairs: JsValue) -> JsValue {
        let terminal_pairs: Vec<(usize, usize)> = pairs.into_serde().unwrap();
        let result = self.inner.solve_for_terminal_pairs(terminal_pairs);
        JsValue::from_serde(&result).unwrap()
    }
}
