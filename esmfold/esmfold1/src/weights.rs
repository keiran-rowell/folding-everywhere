//! In-memory and mmap'd safetensors / PyTorch .bin loader.
//! Returns fp32 `Tensor`s by name, upcasting F16 and BF16 losslessly.


use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

use crate::tensor::Tensor; // adjust path if Tensor is in crate::tensor

pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct Weights<'a> {
    pub(crate) index: HashMap<String, TensorEntry>,
    pub(crate) data: &'a [u8],
}


impl<'a> Weights<'a> {
    fn resolve_key(&self, name: &str) -> Option<&String> {
        if self.index.contains_key(name) {
            return self.index.keys().find(|&k| k == name);
        }

        // Check common aliases
        let aliases: &[(&str, &[&str])] = &[
            (
                "esm.embeddings.word_embeddings.weight",
                &[
                    "esm.embed_tokens.weight",
                    "embed_tokens.weight",
                    "esmfold.esm.embed_tokens.weight",
                    "esm_s.embed_tokens.weight",
                ],
            ),
            (
                "esm.embeddings.position_embeddings.weight",
                &["esm.position_embeddings.weight", "position_embeddings.weight"],
            ),
            (
                "esm.embeddings.LayerNorm.weight",
                &["esm.emb_layer_norm_after.weight", "emb_layer_norm_after.weight"],
            ),
            (
                "esm.embeddings.LayerNorm.bias",
                &["esm.emb_layer_norm_after.bias", "emb_layer_norm_after.bias"],
            ),
        ];

        for (canonical, alts) in aliases {
            if name == *canonical {
                for alt in *alts {
                    if let Some(k) = self.index.keys().find(|&k| k == *alt) {
                        return Some(k);
                    }
                }
            } else if alts.contains(&name) {
                if let Some(k) = self.index.keys().find(|&k| k == *canonical) {
                    return Some(k);
                }
                for alt in *alts {
                    if *alt != name {
                        if let Some(k) = self.index.keys().find(|&k| k == *alt) {
                            return Some(k);
                        }
                    }
                }
            }
        }

        None
    }

    pub fn get(&self, name: &str) -> Tensor {
        let actual_key = match self.resolve_key(name) {
            Some(key) => key,
            None => {
                crate::web_error!("CRITICAL: weight not found: '{name}'");
                panic!("weight not found: {name}");
            }
        };

        let e = &self.index[actual_key];
        // ... rest of your tensor unpacking ...
    }
}
