//! Linear / MatMul operations for Tensors (FP8 + F32)

use crate::tensor::Tensor;
use rayon::prelude::*;

use std::borrow::Cow;

pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let x_owned;
    let x_ref = if x.fp8_bytes.is_some() {
        x_owned = x.to_f32();
        &x_owned
    } else {
        x
    };

    let w_owned;
    let w_ref = if w.fp8_bytes.is_some() {
        w_owned = w.to_f32();
        &w_owned
    } else {
        w
    };

    let b_owned;
    let b_ref = if let Some(bb) = b {
        if bb.fp8_bytes.is_some() {
            b_owned = bb.to_f32();
            Some(&b_owned)
        } else {
            Some(bb)
        }
    } else {
        None
    };

    let xd = &x_ref.data;
    let (m, k) = (x_ref.shape[x_ref.ndim() - 2], x_ref.shape[x_ref.ndim() - 1]);
    let lead: usize = x_ref.numel() / (m * k);
    let mm = lead * m;
    let o = w_ref.shape[0];
    let w_k = if w_ref.ndim() >= 2 { w_ref.shape[1] } else { w_ref.numel() / o };
    if w_k != k {
        crate::web_error!("ops::linear dimension mismatch! x shape = {:?}, k = {}, w shape = {:?}, w_k = {}", x_ref.shape, k, w_ref.shape, w_k);
        assert_eq!(w_k, k, "ops::linear inner dimension mismatch");
    }

    let mut out = vec![0.0f32; mm * o];

    if let Some(w_bytes) = &w.fp8_bytes {
        if w_bytes.len() == o * k {
            let scale = w.scale;
            let b_slice = b_ref.map(|bb| bb.data.as_slice());

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

            let mut shape = x_ref.shape.clone();
            let n = shape.len();
            shape[n - 1] = o;
            return Tensor::new(out, shape);
        }
    }

    // Standard F32 Linear fallback
    let w_data = &w_ref.data;
    let b_data = b_ref.map(|bb| &bb.data);
    let x_f32 = x.to_f32();
    let w_f32 = w.to_f32();
    let b_f32 = b.map(|bb| bb.to_f32());
    let xd = &x_f32.data;

    let (m, k) = (x_f32.shape[x_f32.ndim() - 2], x_f32.shape[x_f32.ndim() - 1]);
    let lead: usize = x_f32.numel() / (m * k);
    let mm = lead * m;
    let o = w_f32.shape[0];
    let w_k = if w_f32.ndim() >= 2 { w_f32.shape[1] } else { w_f32.numel() / o };
    if w_k != k {
        crate::web_error!("ops::linear dimension mismatch! x shape = {:?}, k = {}, w shape = {:?}, w_k = {}", x_f32.shape, k, w_f32.shape, w_k);
        assert_eq!(w_k, k, "ops::linear inner dimension mismatch");
    }

    let mut out = vec![0.0f32; mm * o];

    if let Some(w_bytes) = &w.fp8_bytes {
        if w_bytes.len() == o * k {
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
    }

    // Standard F32 Linear fallback (uses w_f32 which is guaranteed to have valid data)
    let w_data = &w_f32.data;
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
