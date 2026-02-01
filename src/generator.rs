#![allow(dead_code)]
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, TextureView};

/// Parameters passed to procedural texture generation shaders.
///
/// This struct is laid out for GPU uniform buffer compatibility.
/// Shaders should expect this layout at binding 1.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextureParams {
    pub color_primary: [f32; 4],
    pub color_secondary: [f32; 4],
    pub seed: u32,
    pub scale: f32,
    pub roughness: f32,
    pub octaves: f32,
    pub persistence: f32,
    pub lacunarity: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for TextureParams {
    fn default() -> Self {
        Self {
            color_primary: [1.0, 1.0, 1.0, 1.0],
            color_secondary: [0.0, 0.0, 0.0, 1.0],
            seed: 0,
            scale: 1.0,
            roughness: 0.5,
            octaves: 4.0,
            persistence: 0.5,
            lacunarity: 2.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

impl TextureParams {
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    pub fn with_primary_color(mut self, color: [f32; 4]) -> Self {
        self.color_primary = color;
        self
    }

    pub fn with_secondary_color(mut self, color: [f32; 4]) -> Self {
        self.color_secondary = color;
        self
    }

    pub fn with_octaves(mut self, octaves: f32) -> Self {
        self.octaves = octaves;
        self
    }

    pub fn with_persistence(mut self, persistence: f32) -> Self {
        self.persistence = persistence;
        self
    }

    pub fn with_lacunarity(mut self, lacunarity: f32) -> Self {
        self.lacunarity = lacunarity;
        self
    }
}

impl PartialEq for TextureParams {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
            && self.scale.to_bits() == other.scale.to_bits()
            && self.roughness.to_bits() == other.roughness.to_bits()
            && self.octaves.to_bits() == other.octaves.to_bits()
            && self.persistence.to_bits() == other.persistence.to_bits()
            && self.lacunarity.to_bits() == other.lacunarity.to_bits()
            && self.color_primary.iter().zip(&other.color_primary).all(|(a, b)| a.to_bits() == b.to_bits())
            && self.color_secondary.iter().zip(&other.color_secondary).all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for TextureParams {}

impl Hash for TextureParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.seed.hash(state);
        self.scale.to_bits().hash(state);
        self.roughness.to_bits().hash(state);
        self.octaves.to_bits().hash(state);
        self.persistence.to_bits().hash(state);
        self.lacunarity.to_bits().hash(state);
        for c in &self.color_primary {
            c.to_bits().hash(state);
        }
        for c in &self.color_secondary {
            c.to_bits().hash(state);
        }
    }
}

/// Key for cached texture lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureKey {
    pub shader_id: String,
    pub params: TextureParams,
    pub resolution: u32,
}

impl TextureKey {
    pub fn new(shader_id: impl Into<String>, params: TextureParams, resolution: u32) -> Self {
        Self {
            shader_id: shader_id.into(),
            params,
            resolution,
        }
    }
}

struct CachedTexture {
    _texture: wgpu::Texture,
    view: TextureView,
}

struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU-based procedural texture generator.
///
/// Generates textures using WGSL compute shaders. Shaders are automatically
/// loaded from `{shader_dir}/{shader_id.to_lowercase()}.wgsl`.
///
/// Textures are cached by their parameters to avoid redundant computation.
pub struct TextureGenerator {
    device: Device,
    queue: Queue,
    shader_dir: PathBuf,
    pipelines: HashMap<String, ComputePipeline>,
    cache: HashMap<TextureKey, CachedTexture>,
}

impl TextureGenerator {
    /// Create a new texture generator.
    ///
    /// Shaders will be loaded from `{shader_dir}/{material_name.to_lowercase()}.wgsl`.
    pub fn new(device: Device, queue: Queue, shader_dir: PathBuf) -> Self {
        Self {
            device,
            queue,
            shader_dir,
            pipelines: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// Get or create a texture for the given key.
    ///
    /// Returns a view to the generated texture. The texture is cached
    /// and will be reused for identical keys.
    ///
    /// The shader is automatically loaded from `{shader_dir}/{shader_id.to_lowercase()}.wgsl`.
    pub fn get_or_create(&mut self, key: &TextureKey) -> &TextureView {
        if !self.cache.contains_key(key) {
            self.ensure_pipeline(&key.shader_id);
            self.generate(key);
        }
        &self.cache.get(key).expect("texture must exist after generation").view
    }

    /// Clear all cached textures.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Reload all shaders and invalidate caches.
    pub fn reload_shaders(&mut self) {
        self.pipelines.clear();
        self.cache.clear();
    }

    /// Check if a texture is cached.
    pub fn is_cached(&self, key: &TextureKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Get the shader directory.
    pub fn shader_dir(&self) -> &PathBuf {
        &self.shader_dir
    }

    fn ensure_pipeline(&mut self, shader_id: &str) {
        if self.pipelines.contains_key(shader_id) {
            return;
        }

        let shader_path = self.shader_dir.join(format!("{}.wgsl", shader_id.to_lowercase()));
        let shader_source = std::fs::read_to_string(&shader_path)
            .unwrap_or_else(|e| panic!("Failed to read shader {:?}: {}", shader_path, e));

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_id),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} bind group layout", shader_id)),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} pipeline layout", shader_id)),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} compute pipeline", shader_id)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        self.pipelines.insert(shader_id.to_string(), ComputePipeline {
            pipeline,
            bind_group_layout,
        });
    }

    fn generate(&mut self, key: &TextureKey) {
        let pipeline_entry = self.pipelines.get(&key.shader_id).unwrap();

        let size = wgpu::Extent3d {
            width: key.resolution,
            height: key.resolution,
            depth_or_array_layers: 1,
        };
        let mip_count = size.max_mips(wgpu::TextureDimension::D2);

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("procedural texture {}", key.shader_id)),
            size,
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("procedural texture generation"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate texture mips"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline_entry.pipeline);

            let workgroup_size = 8u32;

            for mip in 0..mip_count {
                let mip_w = (key.resolution >> mip).max(1);
                let mip_h = (key.resolution >> mip).max(1);

                let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });

                let uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&key.params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline_entry.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&dst_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                    ],
                });

                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(
                    mip_w.div_ceil(workgroup_size),
                    mip_h.div_ceil(workgroup_size),
                    1,
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.cache.insert(key.clone(), CachedTexture {
            _texture: texture,
            view,
        });
    }
}