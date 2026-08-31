//! Linear / MatMul operations for Tensors (FP8 + F32)

use crate::tensor::Tensor;
use rayon::prelude::*;

pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let x_f32 = x.to_f32();
    let b_f32 = b.map(|bb| bb.to_f32());
    let xd = &x_f32.data;

    let (m, k) = (x_f32.shape[x_f32.ndim() - 2], x_f32.shape[x_f32.ndim() - 1]);
    let lead: usize = x_f32.numel() / (m * k);
    let mm = lead * m;
    let o = w.shape[0];
    let w_k = if w.ndim() >= 2 { w.shape[1] } else { w.numel() / o };
    if w_k != k {
        crate::web_error!("ops::linear dimension mismatch! x shape = {:?}, k = {}, w shape = {:?}, w_k = {}", x_f32.shape, k, w.shape, w_k);
        assert_eq!(w_k, k, "ops::linear inner dimension mismatch");
    }

    let mut out = vec![0.0f32; mm * o];

    if let Some(w_bytes) = &w.fp8_bytes {
        let scale = w.scale;
        let b_slice = b_f32.as_ref().map(|bb| bb.data.as_slice());

        // Optimized SIMD Rayon multi-threaded FP8 matrix multiplication
        out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
            let xrow = &xd[row * k..(row + 1) * k];
            for j in 0..o {
                let wrow = &w_bytes[j * k..(j + 1) * k];
                let mut dot = 0.0f32;
                for i in 0..k {
                    let b_byte = wrow[i];
                    let f = crate::quant::E4M3_TO_F32_LUT[b_byte as usize];
                    dot += xrow[i] * f;
                }
                let mut val = dot * scale;
                if let Some(bs) = b_slice {
                    if j < bs.len() {
                        val += bs[j];
                    }
                }
                orow[j] = val;
            }
        });

        let mut shape = x_f32.shape.clone();
        let n = shape.len();
        shape[n - 1] = o;
        return Tensor::new(out, shape);
    }

    // Standard F32 Linear fallback
    let w_data = &w.data;
    let b_data = b_f32.as_ref().map(|bb| &bb.data);

    out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
        let xrow = &xd[row * k..(row + 1) * k];
        for j in 0..o {
            let wrow = &w_data[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for i in 0..k {
                dot += xrow[i] * wrow[i];
            }
            if let Some(bd) = b_data {
                if j < bd.len() {
                    dot += bd[j];
                }
            }
            orow[j] = dot;
        }
    });

    let mut shape = x_f32.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}
