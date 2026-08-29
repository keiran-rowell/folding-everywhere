//! Browser console logging abstraction.

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
use wasm_bindgen::prelude::*;

#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn error(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn warn(s: &str);
}

#[macro_export]
macro_rules! web_log {
    ($($t:tt)*) => {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        $crate::log::log(&format!($($t)*));
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        println!($($t)*);
    };
}

#[macro_export]
macro_rules! web_error {
    ($($t:tt)*) => {
        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        $crate::log::error(&format!($($t)*));
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        eprintln!($($t)*);
    };
}
