// Ultra-performance 16x16 L1 Tiled & 128-bit Vectorized WGSL Compute Shader for ESMFold1 FP8 Linear MatMul

struct Uniforms {
    in_features: u32,
    out_features: u32,
    batch_size: u32,
    scale: f32,
    has_bias: u32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const E4M3_TO_F32: array<f32, 256> = array<f32, 256>(
    0.000000000e+00f, 1.953125000e-03f, 3.906250000e-03f, 5.859375000e-03f, 7.812500000e-03f, 9.765625000e-03f, 1.171875000e-02f, 1.367187500e-02f, 1.562500000e-02f, 1.757812500e-02f, 1.953125000e-02f, 2.148437500e-02f, 2.343750000e-02f, 2.539062500e-02f, 2.734375000e-02f, 2.929687500e-02f, 3.125000000e-02f, 3.515625000e-02f, 3.906250000e-02f, 4.296875000e-02f, 4.687500000e-02f, 5.078125000e-02f, 5.468750000e-02f, 5.859375000e-02f, 6.250000000e-02f, 7.031250000e-02f, 7.812500000e-02f, 8.593750000e-02f, 9.375000000e-02f, 1.015625000e-01f, 1.093750000e-01f, 1.171875000e-01f, 1.250000000e-01f, 1.406250000e-01f, 1.562500000e-01f, 1.718750000e-01f, 1.875000000e-01f, 2.031250000e-01f, 2.187500000e-01f, 2.343750000e-01f, 2.500000000e-01f, 2.812500000e-01f, 3.125000000e-01f, 3.437500000e-01f, 3.750000000e-01f, 4.062500000e-01f, 4.375000000e-01f, 4.687500000e-01f, 5.000000000e-01f, 5.625000000e-01f, 6.250000000e-01f, 6.875000000e-01f, 7.500000000e-01f, 8.125000000e-01f, 8.750000000e-01f, 9.375000000e-01f, 1.000000000e+00f, 1.125000000e+00f, 1.250000000e+00f, 1.375000000e+00f, 1.500000000e+00f, 1.625000000e+00f, 1.750000000e+00f, 1.875000000e+00f, 2.000000000e+00f, 2.250000000e+00f, 2.500000000e+00f, 2.750000000e+00f, 3.000000000e+00f, 3.250000000e+00f, 3.500000000e+00f, 3.750000000e+00f, 4.000000000e+00f, 4.500000000e+00f, 5.000000000e+00f, 5.500000000e+00f, 6.000000000e+00f, 6.500000000e+00f, 7.000000000e+00f, 7.500000000e+00f, 8.000000000e+00f, 9.000000000e+01f, 1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f, 1.800000000e+01f, 2.000000000e+01f, 2.200000000e+01f, 2.400000000e+01f, 2.600000000e+01f, 2.800000000e+01f, 3.000000000e+01f, 3.200000000e+01f, 3.600000000e+01f, 4.000000000e+01f, 4.400000000e+01f, 4.800000000e+01f, 5.200000000e+01f, 5.600000000e+01f, 6.000000000e+01f, 6.400000000e+01f, 7.200000000e+01f, 8.000000000e+01f, 8.800000000e+01f, 9.600000000e+01f, 1.040000000e+02f, 1.120000000e+02f, 1.200000000e+02f, 1.280000000e+02f, 1.440000000e+02f, 1.600000000e+02f, 1.760000000e+02f, 1.920000000e+02f, 2.080000000e+02f, 2.240000000e+02f, 2.400000000e+02f, 2.560000000e+02f, 2.880000000e+02f, 3.200000000e+02f, 3.520000000e+02f, 3.840000000e+02f, 4.160000000e+02f, 4.480000000e+02f, 0.000000000e+00f, -0.000000000e+00f, -1.953125000e-03f, -3.906250000e-03f, -5.859375000e-03f, -7.812500000e-03f, -9.765625000e-03f, -1.171875000e-02f, -1.367187500e-02f, -1.562500000e-02f, -1.757812500e-02f, -1.953125000e-02f, -2.148437500e-02f, -2.343750000e-02f, -2.539062500e-02f, -2.734375000e-02f, -2.929687500e-02f, -3.125000000e-02f, -3.515625000e-02f, -3.906250000e-02f, -4.296875000e-02f, -4.687500000e-02f, -5.078125000e-02f, -5.468750000e-02f, -5.859375000e-02f, -6.250000000e-02f, -7.031250000e-02f, -7.812500000e-02f, -8.593750000e-02f, -9.375000000e-02f, -1.015625000e-01f, -1.093750000e-01f, -1.171875000e-01f, -1.250000000e-01f, -1.406250000e-01f, -1.562500000e-01f, -1.718750000e-01f, -1.875000000e-01f, -2.031250000e-01f, -2.187500000e-01f, -2.343750000e-01f, -2.500000000e-01f, -2.812500000e-01f, -3.125000000e-01f, -3.437500000e-01f, -3.750000000e-01f, -4.062500000e-01f, -4.375000000e-01f, -4.687500000e-01f, -5.000000000e-01f, -5.625000000e-01f, -6.250000000e-01f, -6.875000000e-01f, -7.500000000e-01f, -8.125000000e-01f, -8.750000000e-01f, -9.375000000e-01f, -1.000000000e+00f, -1.125000000e+00f, -1.250000000e+00f, -1.375000000e+00f, -1.500000000e+00f, -1.625000000e+00f, -1.750000000e+00f, -1.875000000e+00f, -2.000000000e+00f, -2.250000000e+00f, -2.500000000e+00f, -2.750000000e+00f, -3.000000000e+00f, -3.250000000e+00f, -3.500000000e+00f, -3.750000000e+00f, -4.000000000e+00f, -4.500000000e+00f, -5.000000000e+00f, -5.500000000e+00f, -6.000000000e+00f, -6.500000000e+00f, -7.000000000e+00f, -7.500000000e+00f, -8.000000000e+00f, -9.000000000e+01f, -1.000000000e+01f, -1.100000000e+01f, -1.200000000e+01f, -1.300000000e+01f, -1.400000000e+01f, -1.500000000e+01f, -1.600000000e+01f, -1.800000000e+01f, -2.000000000e+01f, -2.200000000e+01f, -2.400000000e+01f, -2.600000000e+01f, -2.800000000e+01f, -3.000000000e+01f, -3.200000000e+01f, -3.600000000e+01f, -4.000000000e+01f, -4.400000000e+01f, -4.800000000e+01f, -5.200000000e+01f, -5.600000000e+01f, -6.000000000e+01f, -6.400000000e+01f, -7.200000000e+01f, -8.000000000e+01f, -8.800000000e+01f, -9.600000000e+01f, -1.040000000e+02f, -1.120000000e+02f, -1.200000000e+02f, -1.280000000e+02f, -1.440000000e+02f, -1.600000000e+02f, -1.760000000e+02f, -1.920000000e+02f, -2.080000000e+02f, -2.240000000e+02f, -2.400000000e+02f, -2.560000000e+02f, -2.880000000e+02f, -3.200000000e+02f, -3.520000000e+02f, -3.840000000e+02f, -4.160000000e+02f, -4.480000000e+02f, 0.000000000e+00f
);

// L1 Workgroup Shared Memory SRAM Tiling
var<workgroup> tileInputs: array<array<f32, 16>, 16>;
var<workgroup> tileWeights: array<array<u32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let batch_idx = global_id.y;
    let lx = local_id.x;
    let ly = local_id.y;

    let in_features = params.in_features;
    let packed_words = in_features / 4u;

    var acc: f32 = 0.0;
    let num_tiles = (packed_words + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_idx = t * 16u + lx;

        // 1. Load 128-bit Vectorized Inputs into L1 Shared Memory SRAM
        if (batch_idx < params.batch_size && (t * 16u + ly) * 4u < in_features) {
            let in_idx = batch_idx * in_features + (t * 16u + ly) * 4u + lx;
            tileInputs[ly][lx] = inputs[in_idx];
        } else {
            tileInputs[ly][lx] = 0.0;
        }

        // 2. Load 128-bit Vectorized Packed FP8 Weights into L1 Shared Memory SRAM
        if (row < params.out_features && k_idx < packed_words) {
            tileWeights[ly][lx] = weights[row * packed_words + k_idx];
        } else {
            tileWeights[ly][lx] = 0u;
        }

        // Synchronize L1 SRAM Cache Barrier across all 256 threads in workgroup
        workgroupBarrier();

        // 3. Compute dot product out of super-fast L1 SRAM
        if (row < params.out_features && batch_idx < params.batch_size) {
            for (var i = 0u; i < 16u; i = i + 1u) {
                let packed_val = tileWeights[ly][i];
                let b0 = packed_val & 0xFFu;
                let b1 = (packed_val >> 8u) & 0xFFu;
                let b2 = (packed_val >> 16u) & 0xFFu;
                let b3 = (packed_val >> 24u) & 0xFFu;

                acc += E4M3_TO_F32[b0] * tileInputs[ly][i * 4u + 0u];
                acc += E4M3_TO_F32[b1] * tileInputs[ly][i * 4u + 1u];
                acc += E4M3_TO_F32[b2] * tileInputs[ly][i * 4u + 2u];
                acc += E4M3_TO_F32[b3] * tileInputs[ly][i * 4u + 3u];
            }
        }

        workgroupBarrier();
    }

    if (row < params.out_features && batch_idx < params.batch_size) {
        var result = acc * params.scale;
        if (params.has_bias == 1u) {
            result += bias[row];
        }
        output[batch_idx * params.out_features + row] = result;
    }
}
