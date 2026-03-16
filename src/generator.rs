#![allow(dead_code)]
//! ## Procedural texture shader contract
//!
//! Procedural textures are generated using WGSL compute shaders.
//! Each shader must follow this binding layout exactly:
//!
//! ### Bind group layout
//! - `@group(0) @binding(0)`
//!   - `texture_storage_2d<rgba8unorm, write>`
//!   - Destination texture for the current mip level
//!
//! - `@group(0) @binding(1)`
//!   - Uniform buffer containing [`TextureParams`]
//!
//! ### Entry point
//! ```wgsl
//! @compute
//! @workgroup_size(8, 8, 1)
//! fn main(@builtin(global_invocation_id) id: vec3<u32>) { ... }
//! ```
//!
//! ### Mip generation
//! The compute shader is dispatched once per mip level.
//! The destination texture view always refers to the current mip only.
//!
//! ### Resolution
//! The shader must infer pixel coordinates from
//! `global_invocation_id.xy` and must not write out of bounds.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, TextureView};

pub const NOTEX_SHADER: &str = r#"
// notex.wgsl - "No Texture" placeholder texture
struct Params {
    color_primary: vec4<f32>,
    color_secondary: vec4<f32>,
    seed: u32,
    scale: f32,
    roughness: f32,
    moisture: f32,
    shadow_strength: f32,
    sheen_strength: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

// 5x7 bitmap font - returns 1.0 if pixel is lit
// Rows are encoded as 5-bit values, bit 4 = leftmost pixel
fn get_pixel(char_id: i32, x: i32, y: i32) -> f32 {
    if (x < 0 || x > 4 || y < 0 || y > 6) { return 0.0; }

    var rows: array<u32, 7>;
    switch char_id {
        case 0: { // N
            rows = array<u32, 7>(17u, 17u, 17u, 19u, 21u, 25u, 17u);
        }
        case 1: { // O
            rows = array<u32, 7>(14u, 17u, 17u, 17u, 17u, 17u, 14u);
        }
        case 2: { // T
            rows = array<u32, 7>(4u, 4u, 4u, 4u, 4u, 4u, 31u);
        }
        case 3: { // E
            rows = array<u32, 7>(31u, 16u, 16u, 30u, 16u, 16u, 31u);
        }
        case 4: { // X
            rows = array<u32, 7>(17u, 10u, 4u, 4u, 4u, 10u, 17u);
        }
        default: {
            return 0.0;
        }
    }

    return f32((rows[y] >> u32(4 - x)) & 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(size.x), f32(size.y));

    // Checkerboard background
    let checker_size = 8.0;
    let cx = i32(floor(uv.x * checker_size));
    let cy = i32(floor(uv.y * checker_size));
    let checker = ((cx + cy) % 2) == 0;
    var bg_color = params.color_primary.rgb;
    if (checker) {
        bg_color *= 0.7;
    }

    // Text layout: "NOTEX" - 5 chars, each 5px wide + 2px spacing
    let text_width = 33.0;  // 5*5 + 4*2
    let text_height = 7.0;
    let char_width = 5.0;
    let char_spacing = 2.0;
    let char_pitch = char_width + char_spacing;

    // Scale text to fit ~60% of texture width
    let scale = f32(min(size.x, size.y)) * 0.6 / text_width;

    // Map UV to text coordinates, centered
    let center = vec2<f32>(0.5, 0.5);
    let text_uv = (uv - center) * vec2<f32>(f32(size.x), f32(size.y)) / scale;

    // Convert to text grid coords with (0,0) at bottom-left of text block
    let tx = text_uv.x + text_width * 0.5;
    let ty = -text_uv.y + text_height * 0.5;  // Flip Y for correct orientation

    // Determine which character and local position
    let char_idx = i32(floor(tx / char_pitch));
    let local_x = i32(tx - f32(char_idx) * char_pitch);
    let local_y = i32(ty);

    // Sample text (only if within character bounds, not spacing)
    var text = 0.0;
    if (local_x < i32(char_width) && char_idx >= 0 && char_idx < 5) {
        text = get_pixel(char_idx, local_x, local_y);
    }

    // Final color
    let color = mix(bg_color, params.color_secondary.rgb, text);

    textureStore(output, vec2<i32>(gid.xy), vec4(color, 1.0));
}
"#;

/// Parameters passed to procedural texture generation shaders.
///
/// This struct is uploaded to the GPU as a uniform buffer and must
/// match the expected WGSL layout exactly.
///
/// ## GPU layout
/// - Bound at `@group(0) @binding(1)`
/// - `#[repr(C)]` and `bytemuck::Pod` for safe GPU transfer
///
/// ## Usage
/// These parameters are typically used for noise-based or procedural
/// texture generation (colors, scale, octaves, etc.).
///
/// Padding fields are included to satisfy GPU alignment requirements.
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
            color_primary: [0.0, 0.0, 0.0, 1.0],   // Black
            color_secondary: [1.0, 0.0, 1.0, 1.0], // Magenta
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
    /// Set the random seed used by the shader.
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }
    /// Set the texture scale or frequency.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
    /// Set the roughness parameter.
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }
    /// Set the primary color.
    pub fn with_primary_color(mut self, color: [f32; 4]) -> Self {
        self.color_primary = color;
        self
    }
    /// Set the secondary color.
    pub fn with_secondary_color(mut self, color: [f32; 4]) -> Self {
        self.color_secondary = color;
        self
    }
    /// Set the number of noise octaves.
    pub fn with_octaves(mut self, octaves: f32) -> Self {
        self.octaves = octaves;
        self
    }
    /// Set the persistence factor between octaves.
    pub fn with_persistence(mut self, persistence: f32) -> Self {
        self.persistence = persistence;
        self
    }
    /// Set the lacunarity factor between octaves.
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

/// Key used for procedural texture caching.
///
/// Textures are uniquely identified by:
/// - Shader ID
/// - Texture parameters
/// - Output resolution
///
/// Identical keys will always reuse the same cached texture.
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
    pub fn notex() -> Self {
        Self {
            shader_id: "notex".to_string(),
            params: TextureParams::default(),
            resolution: 128,
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
/// Textures are generated using WGSL compute shaders and cached
/// to avoid redundant computation.
///
/// ## Shader loading
/// Shaders are automatically loaded from:
/// `{shader_dir}/{shader_id.to_lowercase()}.wgsl`
///
/// ## Caching behavior
/// - Pipelines are cached per shader ID
/// - Generated textures are cached per [`TextureKey`]
/// - Identical keys always return the same texture view
///
/// ## Mip generation
/// All mip levels are generated on the GPU in a single compute pass,
/// with one dispatch per mip level.
pub struct TextureGenerator {
    device: Device,
    queue: Queue,
    shader_dir: PathBuf,
    pipelines: HashMap<String, ComputePipeline>,
    cache: HashMap<TextureKey, CachedTexture>,
}

impl TextureGenerator {
    /// Create a new `TextureGenerator`.
    ///
    /// Procedural texture shaders will be loaded from `{shader_dir}/{material_name.to_lowercase()}.wgsl`.
    pub fn new(device: Device, queue: Queue, shader_dir: PathBuf) -> Self {
        Self {
            device,
            queue,
            shader_dir,
            pipelines: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// Get or generate a procedural texture.
    ///
    /// If a texture matching the given [`TextureKey`] already exists,
    /// it is returned from the cache. Otherwise, the texture is generated
    /// using the associated compute shader.
    ///
    /// ## Shader resolution
    /// The shader is loaded from:
    /// `{shader_dir}/{shader_id.to_lowercase()}.wgsl`
    ///
    /// ## Returns
    /// A [`TextureView`] referencing the generated texture.
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

    /// Reload all procedural texture shaders and invalidate caches.
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

        let shader_module = if shader_id == "notex" {
            self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader_id),
                source: wgpu::ShaderSource::Wgsl(NOTEX_SHADER.into()),
            })
        } else {
            let shader_path = self.shader_dir.join(format!("{}.wgsl", shader_id.to_lowercase()));
            let shader_source = std::fs::read_to_string(&shader_path)
                .unwrap_or_else(|e| panic!("Failed to read shader {:?}: {}", shader_path, e));
            self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader_id),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            })
        };

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