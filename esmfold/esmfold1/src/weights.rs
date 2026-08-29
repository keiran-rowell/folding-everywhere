use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

use crate::tensor::Tensor;

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
   /// Return tensor if present in index, without panicking
    pub fn try_get(&self, name: &str) -> Option<Tensor> {
        if self.index.contains_key(name) {
            // Re-use your existing tensor-building logic or call your parser
            Some(self.get(name))
        } else {
            None
        }
    }

    /// Check if a tensor key exists in the safetensors index
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Try fetching the tensor using multiple candidate keys in order
    pub fn get_any(&self, candidates: &[&str]) -> Tensor {
        for &name in candidates {
            if self.index.contains_key(name) {
                return self.get(name);
            }
        }
        panic!(
            "None of the candidate weight tensors found in index: {:?}",
            candidates
        );
    }

    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("Header too short for safetensors".into());
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err("Header length exceeds buffer".into());
        }

        let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
            .map_err(|e| format!("Invalid UTF-8 header: {e}"))?;

        let parsed: serde_json::Value = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid JSON header: {e}"))?;

        let header_obj = parsed
            .as_object()
            .ok_or_else(|| "Header is not a JSON object".to_string())?;

        let mut index = HashMap::new();
        let data_offset = 8 + header_len;

        for (name, val) in header_obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = val["dtype"].as_str().unwrap_or("F32").to_string();
            let shape = val["shape"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let offsets = val["data_offsets"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();

            if offsets.len() == 2 {
                index.insert(
                    name.clone(),
                    TensorEntry {
                        dtype,
                        shape,
                        start: offsets[0],
                        end: offsets[1],
                    },
                );
            }
        }

        Ok(Weights {
            index,
            data: &bytes[data_offset..],
        })
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &String> {
        self.index.keys()
    }

    pub fn get_shape(&self, name: &str) -> Option<&[usize]> {
        if let Some(entry) = self.index.get(name) {
            return Some(entry.shape.as_slice());
        }
        let resolved = self.resolve_key(name)?;
        self.index.get(resolved).map(|entry| entry.shape.as_slice())
    }

    fn resolve_key(&self, name: &str) -> Option<&String> {
        // 1. Direct exact match
        if self.index.contains_key(name) {
            return self.index.keys().find(|&k| k == name);
        }

        // 2. Explicit top-level aliases
        let aliases: &[(&str, &[&str])] = &[
            (
                "esm.embeddings.word_embeddings.weight",
                &[
                    "esm.encoder.sentence_encoder.embed_tokens.weight",
                    "esm.sentence_encoder.embed_tokens.weight",
                    "esm.embed_tokens.weight",
                    "embed_tokens.weight",
                    "esmfold.esm.embed_tokens.weight",
                    "esm_s.embed_tokens.weight",
                    "model.esm.embed_tokens.weight",
                ],
            ),
            (
                "esm.embeddings.position_embeddings.weight",
                &[
                    "esm.encoder.sentence_encoder.embed_positions.weight",
                    "esm.position_embeddings.weight",
                    "position_embeddings.weight",
                ],
            ),
            (
                "esm.embeddings.LayerNorm.weight",
                &[
                    "esm.encoder.sentence_encoder.emb_layer_norm_after.weight",
                    "esm.emb_layer_norm_after.weight",
                    "emb_layer_norm_after.weight",
                    "esm.encoder.emb_layer_norm_after.weight",
                ],
            ),
            (
                "esm.embeddings.LayerNorm.bias",
                &[
                    "esm.encoder.sentence_encoder.emb_layer_norm_after.bias",
                    "esm.emb_layer_norm_after.bias",
                    "emb_layer_norm_after.bias",
                    "esm.encoder.emb_layer_norm_after.bias",
                ],
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

        // 3. Systematic HuggingFace -> Fairseq layer remapping
        let remapped = name
            .replace("esm.encoder.layer.", "esm.encoder.sentence_encoder.layers.")
            .replace("attention.self.", "self_attn.")
            .replace("attention.output.dense.", "self_attn.out_proj.")
            .replace("output.dense.", "fc2.")
            .replace("intermediate.dense.", "fc1.")
            .replace("LayerNorm.weight", "final_layer_norm.weight")
            .replace("LayerNorm.bias", "final_layer_norm.bias");

        if let Some(k) = self.index.keys().find(|&k| k == &remapped) {
            return Some(k);
        }

        // 4. Fallback: suffix match
        self.index.keys().find(|k| k.ends_with(name))
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

        if e.start >= self.data.len() || e.end > self.data.len() {
            crate::web_error!("CRITICAL: slice [{}..{}] out of bounds", e.start, e.end);
            panic!("slice out of bounds for {actual_key}");
        }

        let bytes = &self.data[e.start..e.end];
        let num_elements: usize = e.shape.iter().product();

        let data: Vec<f32> = match e.dtype.as_str() {
            "F32" => {
                let mut v = vec![0.0f32; num_elements];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr() as *const f32,
                        v.as_mut_ptr(),
                        num_elements,
                    );
                }
                v
            }
            #[cfg(target_arch = "wasm32")]
            "BF16" => {
                let mut v = Vec::with_capacity(num_elements);
                let u16_ptr = bytes.as_ptr() as *const u16;
                let mut i = 0;

                unsafe {
                    let dst_ptr = v.as_mut_ptr() as *mut v128;
                    while i + 8 <= num_elements {
                        let raw_128 = v128_load(u16_ptr.add(i) as *const v128);

                        let f32_low = i32x4_shl(i32x4_extend_low_i16x8(raw_128), 16);
                        let f32_high = i32x4_shl(i32x4_extend_high_i16x8(raw_128), 16);

                        v128_store(dst_ptr.add(i / 4), f32_low);
                        v128_store(dst_ptr.add((i / 4) + 1), f32_high);
                        i += 8;
                    }
                    v.set_len(i);
                }

                while i < num_elements {
                    let raw = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    v.push(f32::from_bits((raw as u32) << 16));
                    i += 1;
                }
                v
            }
            #[cfg(not(target_arch = "wasm32"))]
            "BF16" => {
                let mut v = Vec::with_capacity(num_elements);
                for c in bytes.chunks_exact(2) {
                    let raw = u16::from_le_bytes([c[0], c[1]]);
                    v.push(f32::from_bits((raw as u32) << 16));
                }
                v
            }
            "F16" => {
                let mut v = Vec::with_capacity(num_elements);
                for c in bytes.chunks_exact(2) {
                    let raw = u16::from_le_bytes([c[0], c[1]]);
                    v.push(half::f16::from_le_bytes([c[0], c[1]]).to_f32());
                }
                v
            }
            other => panic!("get() unsupported dtype {other} for {actual_key}"),
        };

        Tensor::new(data, e.shape.clone())
    }

    pub fn get_i64(&self, name: &str) -> Vec<i64> {
        let actual_key = self
            .resolve_key(name)
            .unwrap_or_else(|| panic!("weight not found: {name}"));
        let e = &self.index[actual_key];
        assert_eq!(e.dtype, "I64");

        if e.start >= self.data.len() || e.end > self.data.len() {
            panic!("slice out of bounds for {name}");
        }

        let bytes = &self.data[e.start..e.end];
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}
