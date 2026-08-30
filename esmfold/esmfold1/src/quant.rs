//! FP8 (E4M3 / Float8_e4m3fn) compile-time lookup table and dequantization primitives.

/// Standard IEEE / PyTorch `float8_e4m3fn` decoding table (1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7).
/// NaN/Inf conventions follow the standard e4m3fn spec (no infinities, 0x7F and 0xFF are NaN).
pub static E4M3_TO_F32_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        let byte = i as u8;
        let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (byte >> 3) & 0x0F;
        let mant = byte & 0x07;

        let val = if exp == 0 {
            // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
            sign * 0.015625f32 * (mant as f32 / 8.0f32)
        } else if exp == 0x0F && mant == 0x07 {
            // 0x7F / 0xFF are NaNs in e4m3fn -> treat as 0.0 for safety during inference
            0.0f32
        } else {
            // Normal: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
            let exponent_scale = (1i32 << exp) as f32 / 128.0f32; // 2^(exp - 7)
            sign * exponent_scale * (1.0f32 + (mant as f32 / 8.0f32))
        };

        lut[i] = val;
        i += 1;
    }
    lut
};

/// Fast scalar dequantize using the lookup table
#[inline(always)]
pub fn dequant_e4m3(raw_byte: u8, scale: f32) -> f32 {
    // Direct array indexing into L1 cache
    E4M3_TO_F32_LUT[raw_byte as usize] * scale
}

/// Dequantize a contiguous slice of FP8 bytes directly into an f32 buffer with SIMD
#[inline]
pub fn dequant_slice_e4m3(raw_bytes: &[u8], scale: f32, out: &mut [f32]) {
    assert_eq!(raw_bytes.len(), out.len());
    
    // Modern LLVM automatically vectorizes LUT lookups or table gathers
    for (i, &b) in raw_bytes.iter().enumerate() {
        out[i] = E4M3_TO_F32_LUT[b as usize] * scale;
    }
}
