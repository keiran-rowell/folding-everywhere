//! Linear / MatMul operations for Tensors (FP8 + F32)

use crate::tensor::Tensor;
use rayon::prelude::*;

pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let (m, k) = (x.shape[x.ndim() - 2], x.shape[x.ndim() - 1]);
    let lead: usize = x.numel() / (m * k);
    let mm = lead * m;
    let o = w.shape[0];

    let mut out = vec![0.0f32; mm * o];
    let xd = &x.data;

    if let Some(w_bytes) = &w.fp8_bytes {
        let scale = w.scale;
        let b_slice = b.map(|bb| bb.data.as_slice());

        // Optimized SIMD Rayon multi-threaded FP8 matrix multiplication
        out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
            let xrow = &xd[row * k..(row + 1) * k];
            for j in 0..o {
                let wrow = &w_bytes[j * k..(j + 1) * k];
                let mut dot = 0.0f32;
                for i in 0..k {
                    let b = wrow[i];
                    let f = crate::quant::E4M3_TO_F32_LUT[b as usize];
                    dot += xrow[i] * f;
                }
                let mut val = dot * scale;
                if let Some(bs) = b_slice {
                    val += bs[j];
                }
                orow[j] = val;
            }
        });

        let mut shape = x.shape.clone();
        let n = shape.len();
        shape[n - 1] = o;
        return Tensor::new(out, shape);
    }

    // Standard F32 Linear fallback
    let w_data = &w.data;
    let b_data = b.map(|bb| &bb.data);

    out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
        let xrow = &xd[row * k..(row + 1) * k];
        for j in 0..o {
            let wrow = &w_data[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for i in 0..k {
                dot += xrow[i] * wrow[i];
            }
            if let Some(bd) = b_data {
                dot += bd[j];
            }
            orow[j] = dot;
        }
    });

    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}
