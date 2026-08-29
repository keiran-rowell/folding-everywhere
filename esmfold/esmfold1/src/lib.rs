pub mod constants;
pub mod esm2;
pub mod heads;
pub mod ops;
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
pub mod par_iter;


pub use tensor::Tensor;
pub use weights::Weights;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
use wasm_bindgen::prelude::*;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub fn fold_esmfold1(seq: &str, weight_bytes: &[u8]) -> Result<String, JsValue> {
    let consts = constants::Constants::embedded();
    let weights = weights::Weights::from_bytes(weight_bytes)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse weights: {e}")))?;

    let output = pipeline::fold(&weights, &consts, seq);
    let l = output.l;
    let pdb_str = pdb::to_pdb(&output.atom37.data, &output.plddt.data, &output.aatype, &consts, l);

    Ok(pdb_str)
}
