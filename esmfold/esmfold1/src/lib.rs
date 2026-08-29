pub mod constants;
pub mod esm2;
pub mod heads;
pub mod log;
pub mod ops;
pub mod par_iter;
pub mod parity;
pub mod pdb;
pub mod pipeline;
pub mod pth;
pub mod rigid;
pub mod structure;
pub mod tensor;
pub mod tokenizer;
pub mod trunk;
pub mod weights;

pub use tensor::Tensor;
pub use weights::Weights;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
use wasm_bindgen::prelude::*;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    web_log!("ESMFold1 WASM64 core initialized with panic hooks.");
}

use std::alloc::{alloc, dealloc, Layout};

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub fn alloc_bytes(len: usize) -> *mut u8 {
    let layout = Layout::from_size_align(len, 16).expect("invalid layout");
    unsafe { alloc(layout) }
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub fn dealloc_bytes(ptr: *mut u8, len: usize) {
    let layout = Layout::from_size_align(len, 16).expect("invalid layout");
    unsafe { dealloc(ptr, layout) }
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub fn fold_esmfold1_from_ptr(
    seq: &str,
    weight_ptr: *const u8,
    weight_len: usize,
    progress_fn: Option<js_sys::Function>,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();

    web_log!("Accessing raw pointer: {:p}, len: {} bytes", weight_ptr, weight_len);
    let weight_bytes = unsafe { std::slice::from_raw_parts(weight_ptr, weight_len) };

    web_log!("Loading embedded constants...");
    let consts = constants::Constants::embedded();

    web_log!("Parsing and indexing weight tensors...");
    let weights = weights::Weights::from_bytes(weight_bytes)
        .map_err(|e| JsValue::from_str(&format!("Weight indexing failed: {e}")))?;

    web_log!("Index built. Running pipeline...");
    let this = JsValue::NULL;
    let mut cb = |stage: &str, frac: f32| {
        web_log!("[{:.1}%] {}", frac * 100.0, stage);
        if let Some(ref f) = progress_fn {
            let _ = f.call2(&this, &JsValue::from_str(stage), &JsValue::from_f64(frac as f64));
        }
    };

    let output = pipeline::fold_cb(&weights, &consts, seq, &mut cb);
    let l = output.l;

    let pdb_str = pdb::to_pdb(&output.atom37.data, &output.plddt.data, &output.aatype, &consts, l);
    Ok(pdb_str)
}
