//! Dynamic ESM-2 backbone (supports 650M, 3B, 150M)
//!
//! Pre-LN transformer, erf-GELU FFN, rotary embeddings on q/k.

use crate::ops;
use crate::tensor::Tensor;
use crate::weights::Weights;
use crate::{web_error, web_log};

const HEAD: usize = 64; // ESM-2 standard per-head dimension
const EPS: f32 = 1e-5;
const TOKEN_DROPOUT_SCALE: f32 = 0.88;

fn ln(x: &Tensor, w: &Weights, prefix: &str) -> Tensor {
    ops::layer_norm(x, &w.get(&format!("{prefix}.weight")), &w.get(&format!("{prefix}.bias")), EPS)
}

fn lin(x: &Tensor, w: &Weights, prefix: &str) -> Tensor {
    ops::linear(x, &w.get(&format!("{prefix}.weight")), Some(&w.get(&format!("{prefix}.bias"))))
}

/// Multi-head self-attention with rotary embeddings
fn attention(
    x_ln: &Tensor,
    w: &Weights,
    layer: usize,
    cos: &[f32],
    sin: &[f32],
    l: usize,
    d_model: usize,
    n_heads: usize,
) -> Tensor {
    let p = format!("esm.encoder.layer.{layer}.attention.self");
    let q = lin(x_ln, w, &format!("{p}.query"));
    let k = lin(x_ln, w, &format!("{p}.key"));
    let v = lin(x_ln, w, &format!("{p}.value"));

    // [L, D] -> [H, L, HEAD]
    let mut qh = vec![0.0f32; n_heads * l * HEAD];
    let mut kh = vec![0.0f32; n_heads * l * HEAD];
    let mut vh = vec![0.0f32; n_heads * l * HEAD];

    for li in 0..l {
        for h in 0..n_heads {
            for d in 0..HEAD {
                let idx = (h * l + li) * HEAD + d;
                let src = li * d_model + h * HEAD + d;
                qh[idx] = q.data[src];
                kh[idx] = k.data[src];
                vh[idx] = v.data[src];
            }
        }
    }

    // Query pre-scale by HEAD^-0.5 BEFORE rotary
    let scale = (HEAD as f32).powf(-0.5);
    for x in qh.iter_mut() {
        *x *= scale;
    }
    ops::apply_rotary_inplace(&mut qh, n_heads, l, HEAD, cos, sin);
    ops::apply_rotary_inplace(&mut kh, n_heads, l, HEAD, cos, sin);

    // Per-head attention scores
    let mut ctx = vec![0.0f32; l * d_model];
    for h in 0..n_heads {
        let qbh = &qh[h * l * HEAD..(h + 1) * l * HEAD];
        let kbh = &kh[h * l * HEAD..(h + 1) * l * HEAD];
        let vbh = &vh[h * l * HEAD..(h + 1) * l * HEAD];

        let mut scores = vec![0.0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                let mut s = 0.0f32;
                for d in 0..HEAD {
                    s += qbh[i * HEAD + d] * kbh[j * HEAD + d];
                }
                scores[i * l + j] = s;
            }
        }
        let probs = ops::softmax_last(&Tensor::new(scores, vec![l, l]));

        for i in 0..l {
            for d in 0..HEAD {
                let mut s = 0.0f32;
                for j in 0..l {
                    s += probs.data[i * l + j] * vbh[j * HEAD + d];
                }
                ctx[i * d_model + h * HEAD + d] = s;
            }
        }
    }

    let ctx_t = Tensor::new(ctx, vec![l, d_model]);
    lin(&ctx_t, w, &format!("esm.encoder.layer.{layer}.attention.output.dense"))
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(data, a.shape.clone())
}

pub fn esm2_states(w: &Weights, ids: &[i64]) -> Vec<Tensor> {
    esm2_states_cb(w, ids, &mut |_| {})
}

pub fn esm2_states_cb(w: &Weights, ids: &[i64], prog: &mut dyn FnMut(usize)) -> Vec<Tensor> {
    let l = ids.len();

    // Auto-detect architecture dimensions from weight shapes
    let we = w.get("esm.embeddings.word_embeddings.weight");
    let vocab_size = we.shape[0];
    let d_model = we.shape[1]; // 1280 for 650M, 2560 for 3B
    let n_heads = d_model / HEAD;
    let n_layers = 36;

    web_log!(
        "esm2: detected architecture d_model={}, n_heads={}, L={}",
        d_model,
        n_heads,
        l
    );

    let (cos, sin) = ops::build_cos_sin(l, HEAD);

    let mut emb = vec![0.0f32; l * d_model];
    for (i, &id) in ids.iter().enumerate() {
        if (id as usize) >= vocab_size {
            web_error!("token id {} at pos {} exceeds vocab size {}", id, i, vocab_size);
            panic!("Corrupted token id {} >= {}", id, vocab_size);
        }
        let row = id as usize * d_model;
        for d in 0..d_model {
            emb[i * d_model + d] = we.data[row + d] * TOKEN_DROPOUT_SCALE;
        }
    }

    let mut x = Tensor::new(emb, vec![l, d_model]);
    let mut states: Vec<Tensor> = Vec::with_capacity(n_layers + 1);
    states.push(x.clone());

    for layer in 0..n_layers {
        let lp = format!("esm.encoder.layer.{layer}");

        // Attention sub-block
        let x_ln = ln(&x, w, &format!("{lp}.attention.LayerNorm"));
        let attn = attention(&x_ln, w, layer, &cos, &sin, l, d_model, n_heads);
        x = add(&x, &attn);

        // FFN sub-block
        let y_ln = ln(&x, w, &format!("{lp}.LayerNorm"));
        let up = lin(&y_ln, w, &format!("{lp}.intermediate.dense"));
        let act = ops::gelu_erf(&up);
        let down = lin(&act, w, &format!("{lp}.output.dense"));
        x = add(&x, &down);

        if layer < n_layers - 1 {
            states.push(x.clone());
        } else {
            let last = ln(&x, w, "esm.encoder.emb_layer_norm_after");
            states.push(last);
        }
        prog(layer + 1);
    }

    web_log!("esm2: successfully completed forward pass with {} states", states.len());
    states
}
