// pipelines.rs
#![allow(dead_code)]
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use wgpu::*;
use crate::shader_preprocessing::compile_wgsl;

/// Options required to enable shadow sampling in a render pipeline.
///
/// When provided, the pipeline will expose additional bindings for
/// shadow comparison sampling, following the shader layout described below.
///
/// ## Shader Binding layout (group 0)
/// - `@binding(n + 1)`: comparison sampler
/// - `@binding(n + 2)`: shadow map as `texture_depth_2d_array`
///
/// Where `n` is the number of material textures bound before shadows.
///
/// The sampler **must** be a comparison sampler compatible with
/// depth textures, and the texture view must point to a depth texture
/// array.
#[derive(Clone, Debug)]
pub struct ShadowOptions {
    /// Comparison sampler used for shadow testing.
    pub sampler: Sampler,
    /// Depth texture array containing shadow maps.
    pub view: TextureView,
}

/// Configuration object for creating a render pipeline.
///
/// `PipelineOptions` describes fixed-function and layout-related state
/// used when creating a `wgpu::RenderPipeline`. It is intentionally
/// builder-based to allow concise and readable pipeline setup.
///
/// This type does **not** own shaders. It only defines how vertex data,
/// rasterization, depth testing, MSAA, and render targets are configured.
///
/// ## Shader Binding layout
///
/// The pipeline is expected to follow this binding convention:
///
/// ### Group 0: Material + textures
/// - `@binding(0)`: trilinear sampler
/// - `@binding(1..n)`: material textures as
///   `texture_2d<f32>` or `texture_multisampled_2d<f32>`
/// - `@binding(n + 1)`: (optional) shadow comparison sampler
/// - `@binding(n + 2)`: (optional) shadow map as
///   `texture_depth_2d_array`
///
/// ### Group 1: Uniforms
/// - `@binding(0..m)`: uniform buffers, in the same order as provided
///
/// Any mismatch between shader expectations and these bindings may
/// result in wgpu validation errors.
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    /// Primitive topology used for rasterization.
    pub topology: PrimitiveTopology,

    /// Number of MSAA samples used by the pipeline.
    ///
    /// This **must** match the sample count of the render target.
    pub msaa_samples: u32,

    /// Optional depth-stencil configuration.
    pub depth_stencil: Option<DepthStencilState>,

    /// Vertex buffer layouts consumed by the vertex shader.
    pub vertex_layouts: Vec<VertexBufferLayout<'static>>,

    /// Optional face culling mode.
    pub cull_mode: Option<Face>,

    /// Color target states for the fragment shader outputs.
    ///
    /// The length of this vector must match the number of render targets.
    pub targets: Vec<Option<ColorTargetState>>,

    /// If true, the pipeline is created without a fragment stage.
    ///
    /// Useful for depth-only or shadow-map rendering.
    pub vertex_only: bool,

    /// Optional shadow sampling configuration.
    pub shadow: Option<ShadowOptions>,
}

impl Default for PipelineOptions {
    /// Creates a default pipeline configuration.
    ///
    /// Defaults are chosen to represent a minimal color-rendering pipeline:
    /// - Triangle list topology
    /// - No MSAA
    /// - No depth testing
    /// - No vertex buffers
    /// - No render targets
    /// - Fragment stage enabled
    /// - No shadows
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            targets: vec![],
            vertex_only: false,
            shadow: None,
        }
    }
}


impl PipelineOptions {
    /// Sets the primitive topology used by the pipeline.
    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Sets the MSAA sample count for the pipeline.
    ///
    /// This must match the sample count of the render target view.
    pub fn with_msaa(mut self, samples: u32) -> Self {
        self.msaa_samples = samples;
        self
    }

    /// Enables depth-stencil testing using the provided state.
    pub fn with_depth_stencil(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil = Some(state);
        self
    }

    /// Adds a vertex buffer layout consumed by the vertex shader.
    ///
    /// Layouts are consumed in the order they are added.
    pub fn with_vertex_layout(mut self, layout: VertexBufferLayout<'static>) -> Self {
        self.vertex_layouts.push(layout);
        self
    }

    /// Sets the face culling mode for rasterization.
    pub fn with_cull_mode(mut self, cull: Face) -> Self {
        self.cull_mode = Some(cull);
        self
    }

    /// Adds a color render target to the pipeline.
    ///
    /// The order of targets must match the fragment shader outputs.
    pub fn with_target(mut self, target: ColorTargetState) -> Self {
        self.targets.push(Some(target));
        self
    }

    /// Configures the pipeline as vertex-only.
    ///
    /// This disables the fragment stage entirely and is typically used
    /// for depth-only or shadow-map passes.
    pub fn depth_only(mut self) -> Self {
        self.vertex_only = true;
        self
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DepthStencilKey {
    format: TextureFormat,
    depth_write_enabled: bool,
    depth_compare: CompareFunction,
}

impl From<&DepthStencilState> for DepthStencilKey {
    fn from(d: &DepthStencilState) -> Self {
        Self {
            format: d.format,
            depth_write_enabled: d.depth_write_enabled,
            depth_compare: d.depth_compare,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PipelineKey {
    shader_path: PathBuf,
    layout_hash: u64,
    topology: PrimitiveTopology,
    msaa_samples: u32,
    depth_stencil: Option<DepthStencilKey>,
    cull_mode: Option<Face>,
    depth_only: bool,
    defines_hash: u64,
}

struct ShaderEntry {
    module: ShaderModule,
}

/// Caches render pipelines to avoid redundant creation.
pub struct PipelineCache {
    device: Device,
    shaders: HashMap<PathBuf, ShaderEntry>,
    pipelines: HashMap<PipelineKey, RenderPipeline>,
    pub(crate) uniform_layouts: HashMap<usize, BindGroupLayout>,
}

impl PipelineCache {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            shaders: HashMap::new(),
            pipelines: HashMap::new(),
            uniform_layouts: HashMap::new(),
        }
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Get or create a uniform bind group layout for N uniform buffers.
    pub(crate) fn uniform_layout(&mut self, buffer_count: usize) -> &BindGroupLayout {
        if !self.uniform_layouts.contains_key(&buffer_count) {
            let entries: Vec<BindGroupLayoutEntry> = (0..buffer_count)
                .map(|i| BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();

            let layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("uniform layout ({})", buffer_count)),
                entries: &entries,
            });
            self.uniform_layouts.insert(buffer_count, layout);
        }
        self.uniform_layouts.get(&buffer_count).unwrap()
    }

    /// Create a bind group from uniform buffers.
    pub(crate) fn create_uniform_bind_group(&mut self, buffers: &[&Buffer], label: &str) -> BindGroup {
        let layout = &self.uniform_layout(buffers.len()).clone();
        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    /// Get or create a render pipeline.
    pub(crate) fn get_or_create(
        &mut self,
        shader_path: &Path,
        bind_group_layouts: &[&BindGroupLayout],
        options: &PipelineOptions,
        defines: &HashSet<String>
    ) -> &RenderPipeline {
        let layout_hash = hash_layouts(bind_group_layouts, &options.vertex_layouts);

        let key = PipelineKey {
            shader_path: shader_path.to_path_buf(),
            layout_hash,
            topology: options.topology,
            msaa_samples: options.msaa_samples,
            depth_stencil: options.depth_stencil.as_ref().map(|d| d.into()),
            cull_mode: options.cull_mode,
            depth_only: options.vertex_only,
            defines_hash: defines.hasher().build_hasher().finish()
        };

        if !self.pipelines.contains_key(&key) {
            self.load_shader(shader_path, defines);
            let pipeline = self.create_pipeline(&key, bind_group_layouts, options);
            self.pipelines.insert(key.clone(), pipeline);
        }

        self.pipelines.get(&key).unwrap()
    }

    /// Reload shaders from disk. Pipelines using reloaded shaders will be recreated on next use.
    pub(crate) fn reload_shaders(&mut self, paths: &[PathBuf], variables: &HashSet<String>) {
        for path in paths {
            if self.shaders.contains_key(path) {
                self.load_shader(path, variables);
            }
        }
        self.pipelines.retain(|key, _| !paths.contains(&key.shader_path));
    }

    /// Clear all cached pipelines and shaders.
    pub(crate) fn clear(&mut self) {
        self.shaders.clear();
        self.pipelines.clear();
    }

    fn load_shader(&mut self, path: &Path, defines: &HashSet<String>) {
        let module = compile_wgsl(&self.device, path, defines);

        self.shaders.insert(path.to_path_buf(), ShaderEntry { module });
    }

    fn create_pipeline(
        &self,
        key: &PipelineKey,
        bind_group_layouts: &[&BindGroupLayout],
        options: &PipelineOptions,
    ) -> RenderPipeline {
        let shader = &self.shaders.get(&key.shader_path).unwrap().module;

        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} layout", key.shader_path.display())),
            bind_group_layouts,
            immediate_size: 0,
        });

        let fragment = if options.vertex_only {
            None
        } else {
            Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &options.targets.iter().cloned().collect::<Vec<_>>(),
                compilation_options: Default::default(),
            })
        };

        self.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(&format!("{} Pipeline", key.shader_path.display())),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &options.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment,
            primitive: PrimitiveState {
                topology: options.topology,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: options.cull_mode,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: options.depth_stencil.clone(),
            multisample: MultisampleState {
                count: options.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        })
    }
}

fn hash_layouts(bgls: &[&BindGroupLayout], vertex_layouts: &[VertexBufferLayout]) -> u64 {
    let mut hasher = DefaultHasher::new();

    for bgl in bgls {
        (*bgl as *const BindGroupLayout as usize).hash(&mut hasher);
    }

    for layout in vertex_layouts {
        layout.array_stride.hash(&mut hasher);
        (layout.step_mode as u32).hash(&mut hasher);
        for attr in layout.attributes {
            attr.shader_location.hash(&mut hasher);
            attr.offset.hash(&mut hasher);
            (attr.format as u32).hash(&mut hasher);
        }
    }

    hasher.finish()
}
