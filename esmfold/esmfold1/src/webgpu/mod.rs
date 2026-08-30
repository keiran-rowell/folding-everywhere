//! WebGPU Compute Engine for ESMFold1 FP8 operations with 0-allocation Persistent Scratchpad Buffers

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// Uniforms buffer structure matching WGSL shader
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulUniforms {
    pub in_features: u32,
    pub out_features: u32,
    pub batch_size: u32,
    pub scale: f32,
    pub has_bias: u32,
    pub _padding: [u32; 3],
}

pub struct WebGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub weight_buffer_cache: Mutex<HashMap<usize, Arc<wgpu::Buffer>>>,
    pub scratch_input: wgpu::Buffer,
    pub scratch_output: wgpu::Buffer,
    pub scratch_uniforms: wgpu::Buffer,
    pub scratch_bias: wgpu::Buffer,
}

pub struct SendSyncContext(pub Arc<WebGpuContext>);
unsafe impl Send for SendSyncContext {}
unsafe impl Sync for SendSyncContext {}

pub static GLOBAL_WEBGPU: Mutex<Option<SendSyncContext>> = Mutex::new(None);

impl WebGpuContext {
    /// Initialize WebGPU context asynchronously with persistent 64MB scratchpad VRAM buffers
    pub async fn init() -> Option<Arc<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ESMFold1 WebGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let shader_source = include_str!("matmul_fp8.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ESMFold1 Matmul FP8 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul FP8 Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul FP8 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul FP8 Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pre-allocate persistent scratchpad buffers (64MB capacity) to eliminate driver allocation overhead
        let scratch_size = (64 * 1024 * 1024) as u64;

        let scratch_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent Input Scratch Buffer"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent Output Scratch Buffer"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent Uniforms Buffer"),
            size: std::mem::size_of::<MatmulUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_bias = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent Bias Scratch Buffer"),
            size: (16 * 1024 * 1024) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ctx = Arc::new(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            weight_buffer_cache: Mutex::new(HashMap::new()),
            scratch_input,
            scratch_output,
            scratch_uniforms,
            scratch_bias,
        });

        if let Ok(mut lock) = GLOBAL_WEBGPU.lock() {
            *lock = Some(SendSyncContext(ctx.clone()));
        }

        Some(ctx)
    }

    /// Upload & Cache Weight Matrix in GPU Memory
    pub fn get_or_create_weight_buffer(&self, fp8_weights: &[u8]) -> Arc<wgpu::Buffer> {
        let weight_ptr_addr = fp8_weights.as_ptr() as usize;
        if let Ok(mut cache) = self.weight_buffer_cache.lock() {
            if let Some(buf) = cache.get(&weight_ptr_addr) {
                return buf.clone();
            }
            let u32_weights: &[u32] = bytemuck::cast_slice(fp8_weights);
            let buf = Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matmul Weights Buffer"),
                contents: bytemuck::cast_slice(u32_weights),
                usage: wgpu::BufferUsages::STORAGE,
            }));
            cache.insert(weight_ptr_addr, buf.clone());
            buf
        } else {
            let u32_weights: &[u32] = bytemuck::cast_slice(fp8_weights);
            Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matmul Weights Buffer"),
                contents: bytemuck::cast_slice(u32_weights),
                usage: wgpu::BufferUsages::STORAGE,
            }))
        }
    }

    /// Dispatch FP8 Linear MatMul with 0-allocation queue writes
    pub async fn dispatch_matmul_fp8(
        &self,
        input: &[f32],
        fp8_weights: &[u8],
        scale: f32,
        bias: Option<&[f32]>,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        let has_bias = if bias.is_some() { 1u32 } else { 0u32 };

        let uniforms = MatmulUniforms {
            in_features: in_features as u32,
            out_features: out_features as u32,
            batch_size: batch_size as u32,
            scale,
            has_bias,
            _padding: [0; 3],
        };

        // Write to persistent GPU scratchpad buffers (0 allocation overhead!)
        self.queue.write_buffer(&self.scratch_uniforms, 0, bytemuck::cast_slice(&[uniforms]));
        self.queue.write_buffer(&self.scratch_input, 0, bytemuck::cast_slice(input));

        if let Some(b) = bias {
            self.queue.write_buffer(&self.scratch_bias, 0, bytemuck::cast_slice(b));
        }

        let weight_buffer = self.get_or_create_weight_buffer(fp8_weights);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul FP8 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scratch_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scratch_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scratch_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.scratch_output.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Command Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = ((out_features as u32) + 15) / 16;
            let workgroups_y = ((batch_size as u32) + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

#[derive(Clone)]
pub struct GpuTensor {
    pub buffer: Arc<wgpu::Buffer>,
    pub shape: Vec<usize>,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

impl WebGpuContext {
    pub fn upload_to_gpu(&self, t: &crate::tensor::Tensor) -> GpuTensor {
        let buffer = Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuTensor Activation Buffer"),
            contents: bytemuck::cast_slice(&t.data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }));
        GpuTensor {
            buffer,
            shape: t.shape.clone(),
        }
    }

    pub fn dispatch_matmul_fp8_in_vram(
        &self,
        input: &GpuTensor,
        fp8_weights: &[u8],
        scale: f32,
        bias: Option<&[f32]>,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<GpuTensor, String> {
        let has_bias = if bias.is_some() { 1u32 } else { 0u32 };

        let uniforms = MatmulUniforms {
            in_features: in_features as u32,
            out_features: out_features as u32,
            batch_size: batch_size as u32,
            scale,
            has_bias,
            _padding: [0; 3],
        };

        self.queue.write_buffer(&self.scratch_uniforms, 0, bytemuck::cast_slice(&[uniforms]));
        if let Some(b) = bias {
            self.queue.write_buffer(&self.scratch_bias, 0, bytemuck::cast_slice(b));
        }

        let weight_buffer = self.get_or_create_weight_buffer(fp8_weights);
        let out_bytes = (batch_size * out_features * std::mem::size_of::<f32>()) as u64;

        let out_buffer = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor Output Activation Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul FP8 In-VRAM Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scratch_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scratch_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul In-VRAM Command Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul In-VRAM Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = ((out_features as u32) + 15) / 16;
            let workgroups_y = ((batch_size as u32) + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        let mut out_shape = input.shape.clone();
        let n = out_shape.len();
        out_shape[n - 1] = out_features;

        Ok(GpuTensor {
            buffer: out_buffer,
            shape: out_shape,
        })
    }

    pub async fn readback_to_cpu(&self, gpu_t: &GpuTensor) -> Result<crate::tensor::Tensor, String> {
        let size = (gpu_t.numel() * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Readback Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Command Encoder"),
        });
        encoder.copy_buffer_to_buffer(&gpu_t.buffer, 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result_slice: &[f32] = bytemuck::cast_slice(&data);
            let vec_data = result_slice.to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(crate::tensor::Tensor::new(vec_data, gpu_t.shape.clone()))
        } else {
            Err("Failed to map staging buffer during readback".to_string())
        }
    }

}

/// High-performance Command Buffer Chaining for multi-pass compute pipelines
pub struct CommandBatcher<'a> {
    pub ctx: &'a WebGpuContext,
    pub encoder: wgpu::CommandEncoder,
    pub pass_count: usize,
}

impl<'a> CommandBatcher<'a> {
    pub fn new(ctx: &'a WebGpuContext, label: &str) -> Self {
        let encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(label),
        });
        Self {
            ctx,
            encoder,
            pass_count: 0,
        }
    }

    /// Chain a MatMul pass without submitting to GPU queue
    pub fn record_matmul(
        &mut self,
        fp8_weights: &[u8],
        scale: f32,
        bias: Option<&[f32]>,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
    ) {
        let has_bias = if bias.is_some() { 1u32 } else { 0u32 };

        let uniforms = MatmulUniforms {
            in_features: in_features as u32,
            out_features: out_features as u32,
            batch_size: batch_size as u32,
            scale,
            has_bias,
            _padding: [0; 3],
        };

        self.ctx.queue.write_buffer(&self.ctx.scratch_uniforms, 0, bytemuck::cast_slice(&[uniforms]));
        if let Some(b) = bias {
            self.ctx.queue.write_buffer(&self.ctx.scratch_bias, 0, bytemuck::cast_slice(b));
        }

        let weight_buffer = self.ctx.get_or_create_weight_buffer(fp8_weights);

        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chained Matmul Bind Group"),
            layout: &self.ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.ctx.scratch_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.ctx.scratch_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ctx.scratch_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.ctx.scratch_output.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Chained Matmul Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = ((out_features as u32) + 15) / 16;
            let workgroups_y = ((batch_size as u32) + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        self.pass_count += 1;
    }

    /// Submit chained command buffer to GPU queue in a single atomic dispatch
    pub fn submit(self) {
        if self.pass_count > 0 {
            self.ctx.queue.submit(Some(self.encoder.finish()));
        }
    }
}
