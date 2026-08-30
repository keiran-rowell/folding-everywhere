#![cfg_attr(target_arch = "wasm64", feature(simd_wasm64))]

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
pub mod quant;
pub mod rigid;
pub mod structure;
pub mod tensor;
pub mod tokenizer;
pub mod trunk;
pub mod webgpu;
pub mod weights;

pub use tensor::Tensor;
pub use weights::Weights;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
use wasm_bindgen::prelude::*;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    web_log!("ESMFold1 WASM core initialized with panic hooks.");
}

#[cfg(target_arch = "wasm64")]
use std::arch::wasm64 as wasm_arch;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32 as wasm_arch;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub fn alloc_bytes(len: u64) -> *mut u8 {
    let size = len as usize;
    web_log!("alloc_bytes: allocating {} bytes ({} MB)...", size, size / (1024 * 1024));

    let pages_needed = (size + 65535) / 65536;
    let prev_pages = wasm_arch::memory_grow(0, pages_needed);
    
    if prev_pages == usize::MAX {
        web_error!("CRITICAL: wasm memory.grow failed for {} pages", pages_needed);
        return std::ptr::null_mut();
    }

    let ptr = (prev_pages * 65536) as *mut u8;
    web_log!("alloc_bytes: allocated buffer via memory.grow at {:p}", ptr);
    ptr
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub fn dealloc_bytes(ptr: *mut u8, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
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

    web_log!("Step 1: Validating pointer {:p} (len: {} bytes)...", weight_ptr, weight_len);

    if weight_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed to fold_esmfold1_from_ptr"));
    }

    let weight_bytes = unsafe { std::slice::from_raw_parts(weight_ptr, weight_len) };

    web_log!("Step 2: Verifying slice magic bytes: {:02X?}", &weight_bytes[..4.min(weight_bytes.len())]);
    web_log!("Step 3: Loading embedded constants...");
    let consts = constants::Constants::embedded();

    web_log!("Step 4: Indexing weight buffer...");
    let weights = match weights::Weights::from_bytes(weight_bytes) {
        Ok(w) => w,
        Err(e) => {
            let err_str = format!("Weights parsing error: {e}");
            web_error!("{err_str}");
            return Err(JsValue::from_str(&err_str));
        }
    };

    web_log!("Step 5: Found {} tensors. Executing forward pass...", weights.names().len());

    let this = JsValue::NULL;
    let mut cb = |stage: &str, frac: f32| {
        web_log!("[{:.1}%] {}", frac * 100.0, stage);
        if let Some(ref f) = progress_fn {
            let _ = f.call2(&this, &JsValue::from_str(stage), &JsValue::from_f64(frac as f64));
        }
    };

    let output = pipeline::fold_cb(&weights, &consts, seq, &mut cb);
    let l = output.l;

    web_log!("Step 6: Reconstructing PDB...");
    let pdb_str = pdb::to_pdb(&output.atom37.data, &output.plddt.data, &output.aatype, &consts, l);
    web_log!("Fold complete. PDB length: {} bytes.", pdb_str.len());

    Ok(pdb_str)
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub async fn init_webgpu_backend() -> bool {
    let fut = webgpu::WebGpuContext::init();
    match fut.await {
        Some(_ctx) => {
            web_log!("WebGPU backend initialized successfully!");
            true
        }
        None => {
            web_log!("WebGPU unavailable on device/browser. Using multi-threaded SIMD CPU fallback.");
            false
        }
    }
}

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
pub async fn fold_esmfold1_from_ptr_async(
    seq: &str,
    weight_ptr: *const u8,
    weight_len: usize,
    progress_fn: Option<js_sys::Function>,
) -> Result<String, JsValue> {
    fold_esmfold1_from_ptr(seq, weight_ptr, weight_len, progress_fn)
}
