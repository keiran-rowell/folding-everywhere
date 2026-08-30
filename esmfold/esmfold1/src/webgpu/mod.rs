//! WebGPU Compute Engine for ESMFold1 FP8 operations

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
}

pub struct SendSyncContext(pub Arc<WebGpuContext>);
unsafe impl Send for SendSyncContext {}
unsafe impl Sync for SendSyncContext {}

pub static GLOBAL_WEBGPU: Mutex<Option<SendSyncContext>> = Mutex::new(None);

impl WebGpuContext {
    /// Initialize WebGPU context asynchronously
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

        let ctx = Arc::new(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            weight_buffer_cache: Mutex::new(HashMap::new()),
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

    /// Dispatch FP8 Linear MatMul directly to WebGPU WGSL Compute Pipeline
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

        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matmul Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matmul Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let weight_buffer = self.get_or_create_weight_buffer(fp8_weights);

        let dummy_bias = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy_bias);
        let bias_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matmul Bias Buffer"),
            contents: bytemuck::cast_slice(bias_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer_size = (batch_size * out_features * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matmul Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matmul Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul FP8 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
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

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Poll);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let result_slice: &[f32] = bytemuck::cast_slice(&data);
                output.copy_from_slice(result_slice);
                drop(data);
                staging_buffer.unmap();
                Ok(())
            }
            _ => Err("Failed to map WebGPU staging buffer".to_string()),
        }
    }
}
