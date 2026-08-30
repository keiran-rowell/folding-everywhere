//! Matmul / Linear with a pinned accumulation order.
//!
//! Parallelism (rayon) is over OUTPUT ROWS only; the K-reduction for any single
//! output element is always a sequential fp32 fold in ascending k. Therefore the
//! result is independent of thread count -> deterministic.

use crate::quant::E4M3_TO_F32_LUT;
use crate::tensor::Tensor;
use crate::par_iter::*;

/// Vectorizable fp32 dot product with a fixed 8-lane partial-sum order.
#[inline]
pub fn dot8(a: &[f32], b: &[f32], k: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let nchunks = k / 8;
    for c in 0..nchunks {
        let ai = &a[c * 8..c * 8 + 8];
        let bi = &b[c * 8..c * 8 + 8];
        for l in 0..8 {
            acc[l] += ai[l] * bi[l];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for kk in (nchunks * 8)..k {
        s += a[kk] * b[kk];
    }
    s
}

/// PyTorch-style Linear: x[M,K] @ w[O,K]^T + b[O]  ->  [M,O].
/// `w` is row-major [O,K] (the torch `nn.Linear.weight` layout).
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Tensor {
    let (m, k) = (x.shape[x.ndim() - 2], x.shape[x.ndim() - 1]);
    let lead: usize = x.numel() / (m * k); // any leading batch dims collapsed
    let mm = lead * m;
    let o = w.shape[0];
    assert_eq!(w.shape[1], k, "linear K mismatch x{:?} w{:?}", x.shape, w.shape);
    if let Some(bb) = b {
        assert_eq!(bb.numel(), o);
    }
    let mut out = vec![0.0f32; mm * o];
    let xd = &x.data;

    if let Some(w_bytes) = &w.fp8_bytes {
        let scale = w.scale;
        let b_slice = b.map(|bb| bb.data.as_slice());

        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            if let Ok(lock) = crate::webgpu::GLOBAL_WEBGPU.lock() {
                if let Some(ref send_sync_ctx) = *lock {
                    let gpu = &send_sync_ctx.0;
                    let fut = gpu.dispatch_matmul_fp8(
                        xd,
                        w_bytes,
                        scale,
                        b_slice,
                        mm,
                        k,
                        o,
                        &mut out,
                    );
                    if pollster::block_on(fut).is_ok() {
                        let mut shape = x.shape.clone();
                        let n = shape.len();
                        shape[n - 1] = o;
                        return Tensor::new(out, shape);
                    }
                }
            }
        }

        let scale = w.scale;
        out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
            let xrow = &xd[row * k..row * k + k];
            let mut wrow_f32 = vec![0.0f32; k];
            for oi in 0..o {
                let w_fp8 = &w_bytes[oi * k..oi * k + k];
                for col in 0..k {
                    wrow_f32[col] = E4M3_TO_F32_LUT[w_fp8[col] as usize] * scale;
                }
                let d = dot8(xrow, &wrow_f32, k);
                orow[oi] = if let Some(bb) = b {
                    d + bb.data[oi]
                } else {
                    d
                };
            }
        });
    } else {
        let wd = &w.data;
        out.par_chunks_mut(o).enumerate().for_each(|(row, orow)| {
            let xrow = &xd[row * k..row * k + k];
            for oi in 0..o {
                let d = dot8(xrow, &wd[oi * k..oi * k + k], k);
                orow[oi] = if let Some(bb) = b {
                    d + bb.data[oi]
                } else {
                    d
                };
            }
        });
    }

    let mut shape = x.shape.clone();
    let n = shape.len();
    shape[n - 1] = o;
    Tensor::new(out, shape)
}

/// General 2D matmul a[M,K] @ b[K,N] -> [M,N], sequential-K fp32 accumulation.
pub fn matmul2d(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    assert_eq!(b.ndim(), 2);
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2);
    let ad = &a.data;
    let bd = &b.data;
    let mut out = vec![0.0f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
        let arow = &ad[i * k..i * k + k];
        for kk in 0..k {
            let aik = arow[kk];
            let brow = &bd[kk * n..kk * n + n];
            for j in 0..n {
                orow[j] += aik * brow[j];
            }
        }
    });
    Tensor::new(out, vec![m, n])
}
