//! In-memory and mmap'd safetensors / PyTorch .bin loader.
//! Returns fp32 `Tensor`s by name, upcasting F16 and BF16 losslessly.

use crate::tensor::Tensor;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize, // ABSOLUTE byte offset in the buffer
    end: usize,
}

pub enum WeightData<'a> {
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    Mmap(memmap2::Mmap),
    Borrowed(&'a [u8]),
}

impl<'a> std::ops::Deref for WeightData<'a> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        match self {
            #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
            WeightData::Mmap(m) => m,
            WeightData::Borrowed(b) => b,
        }
    }
}

pub struct Weights<'a> {
    data: WeightData<'a>,
    index: HashMap<String, Entry>,
}

impl<'a> Weights<'a> {
    /// Native disk loader: memory-maps a file from disk (CLI / desktop only).
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    pub fn open(path: &str) -> std::io::Result<Weights<'static>> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let index = Self::build_index(&mmap)?;
        Ok(Weights {
            data: WeightData::Mmap(mmap),
            index,
        })
    }

    /// In-memory buffer loader: accepts raw bytes directly (WASM / Web Worker).
    pub fn from_bytes(bytes: &'a [u8]) -> std::io::Result<Self> {
        let index = Self::build_index(bytes)?;
        Ok(Weights {
            data: WeightData::Borrowed(bytes),
            index,
        })
    }

    /// Auto-detects PyTorch ZIP vs. Safetensors format and builds the tensor index.
    fn build_index(buf: &[u8]) -> std::io::Result<HashMap<String, Entry>> {
        if buf.len() >= 4 && &buf[0..4] == b"PK\x03\x04" {
            // PyTorch .bin (ZIP archive)
            Ok(crate::pth::index_pth(buf)
                .into_iter()
                .map(|e| (e.name, Entry { dtype: e.dtype, shape: e.shape, start: e.start, end: e.end }))
                .collect())
        } else {
            // Safetensors
            let header_len = u64::from_le_bytes(
                buf[0..8].try_into().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            ) as usize;

            let json: serde_json::Value = serde_json::from_slice(&buf[8..8 + header_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let data_start = 8 + header_len;
            let mut index = HashMap::new();

            if let serde_json::Value::Object(map) = json {
                for (k, v) in map {
                    if k == "__metadata__" {
                        continue;
                    }
                    let dtype = v["dtype"].as_str().unwrap().to_string();
                    let shape: Vec<usize> = v["shape"].as_array().unwrap().iter()
                        .map(|x| x.as_u64().unwrap() as usize).collect();
                    let offs = v["data_offsets"].as_array().unwrap();
                    let start = data_start + offs[0].as_u64().unwrap() as usize;
                    let end = data_start + offs[1].as_u64().unwrap() as usize;
                    index.insert(k, Entry { dtype, shape, start, end });
                }
            }
            Ok(index)
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|e| e.shape.as_slice())
    }

    /// Fetch a tensor as fp32 (F16 and BF16 upcast losslessly). Panics if name missing.
    pub fn get(&self, name: &str) -> Tensor {
        let e = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("weight not found: {name}"));
        let bytes = &self.data[e.start..e.end];
        let data: Vec<f32> = match e.dtype.as_str() {
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => panic!("get() unsupported dtype {other} for {name}"),
        };
        Tensor::new(data, e.shape.clone())
    }

    pub fn get_i64(&self, name: &str) -> Vec<i64> {
        let e = self.index.get(name).unwrap_or_else(|| panic!("weight not found: {name}"));
        assert_eq!(e.dtype, "I64");
        let bytes = &self.data[e.start..e.end];
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}
