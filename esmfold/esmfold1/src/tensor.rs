//! Minimal row-major, contiguous, fp32/fp8 tensor.

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub fp8_bytes: Option<Vec<u8>>,
    pub scale: f32,
}

impl Tensor {
    /// Standard FP32 Tensor constructor
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "shape {:?} ({}) != data len {}",
            shape,
            n,
            data.len()
        );
        Self {
            data,
            shape,
            fp8_bytes: None,
            scale: 1.0,
        }
    }

    /// Construct a tensor holding quantized FP8 bytes
    pub fn new_fp8(bytes: Vec<u8>, shape: Vec<usize>, scale: f32) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            bytes.len(),
            "FP8 shape {:?} ({}) != bytes len {}",
            shape,
            n,
            bytes.len()
        );
        Self {
            data: Vec::new(),
            shape,
            fp8_bytes: Some(bytes),
            scale,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0f32; size],
            shape: shape.to_vec(),
            fp8_bytes: None,
            scale: 1.0,
        }
    }

    pub fn filled(shape: &[usize], v: f32) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![v; size],
            shape: shape.to_vec(),
            fp8_bytes: None,
            scale: 1.0,
        }
    }

    #[inline]
    pub fn numel(&self) -> usize {
        if let Some(ref b) = self.fp8_bytes {
            b.len()
        } else {
            self.data.len()
        }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape in place (must preserve element count).
    pub fn reshape(mut self, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.numel(), "reshape {:?} -> {:?}", self.shape, shape);
        self.shape = shape.to_vec();
        self
    }

    /// Explicitly dequantize an FP8 tensor to an FP32 Tensor (or clone if already FP32)
    pub fn to_f32(&self) -> Tensor {
        if let Some(ref bytes) = self.fp8_bytes {
            let mut out = vec![0.0f32; bytes.len()];
            crate::quant::dequant_slice_e4m3(bytes, self.scale, &mut out);
            Tensor::new(out, self.shape.clone())
        } else {
            self.clone()
        }
    }
}
