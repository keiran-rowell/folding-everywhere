//! LayerNorm and softmax over the last dimension, fp32, pinned reduction order.

use crate::tensor::Tensor;

/// LayerNorm over the last dim. Biased variance, eps inside sqrt (matches ATen).
/// `weight`/`bias` have length C (the last dim). Two-pass mean/var in fp32.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let c = x.shape[x.ndim() - 1];
    let weight_f32 = weight.to_f32();
    let bias_f32 = bias.to_f32();
    if weight_f32.numel() != c || bias_f32.numel() != c {
        crate::web_error!("layer_norm shape mismatch: x shape = {:?}, C = {}, weight numel = {}, bias numel = {}", x.shape, c, weight_f32.numel(), bias_f32.numel());
    }
    assert_eq!(weight_f32.numel(), c);
    assert_eq!(bias_f32.numel(), c);
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    let w = &weight_f32.data;
    let b = &bias_f32.data;
    for r in 0..rows {
        let xr = &x.data[r * c..r * c + c];
        let or = &mut out[r * c..r * c + c];
        let mut mean = 0.0f32;
        for &v in xr {
            mean += v;
        }
        mean /= c as f32;
        let mut var = 0.0f32;
        for &v in xr {
            let d = v - mean;
            var += d * d;
        }
        var /= c as f32;
        let rstd = 1.0f32 / (var + eps).sqrt();
        for i in 0..c {
            or[i] = (xr[i] - mean) * rstd * w[i] + b[i];
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// Softmax over the last dimension (max-subtracted, libm exp).
pub fn softmax_last(x: &Tensor) -> Tensor {
    let c = x.shape[x.ndim() - 1];
    let rows = x.numel() / c;
    let mut out = vec![0.0f32; x.numel()];
    for r in 0..rows {
        let xr = &x.data[r * c..r * c + c];
        let or = &mut out[r * c..r * c + c];
        let mut m = f32::NEG_INFINITY;
        for &v in xr {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f32;
        for i in 0..c {
            let e = libm::expf(xr[i] - m);
            or[i] = e;
            s += e;
        }
        let inv = 1.0f32 / s;
        for i in 0..c {
            or[i] *= inv;
        }
    }
    Tensor::new(out, x.shape.clone())
}
