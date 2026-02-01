// pipelines.rs
#![allow(dead_code)]
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use wgpu::*;

#[derive(Clone, Debug)]
pub struct ShadowOptions {
    pub sampler: Sampler,
    pub view: TextureView,
}

/// Configuration for creating a render pipeline.
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    pub topology: PrimitiveTopology,
    pub msaa_samples: u32,
    pub depth_stencil: Option<DepthStencilState>,
    pub vertex_layouts: Vec<VertexBufferLayout<'static>>,
    pub cull_mode: Option<Face>,
    pub targets: Vec<Option<ColorTargetState>>,
    pub vertex_only: bool,
    pub shadow: Option<ShadowOptions>
}

impl Default for PipelineOptions {
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
    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn with_msaa(mut self, samples: u32) -> Self {
        self.msaa_samples = samples;
        self
    }

    pub fn with_depth_stencil(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil = Some(state);
        self
    }

    pub fn with_vertex_layout(mut self, layout: VertexBufferLayout<'static>) -> Self {
        self.vertex_layouts.push(layout);
        self
    }

    pub fn with_cull_mode(mut self, cull: Face) -> Self {
        self.cull_mode = Some(cull);
        self
    }

    pub fn with_target(mut self, target: ColorTargetState) -> Self {
        self.targets.push(Some(target));
        self
    }

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

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get or create a uniform bind group layout for N uniform buffers.
    pub fn uniform_layout(&mut self, buffer_count: usize) -> &BindGroupLayout {
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
    pub fn create_uniform_bind_group(&mut self, buffers: &[&Buffer], label: &str) -> BindGroup {
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
    pub fn get_or_create(
        &mut self,
        shader_path: &Path,
        bind_group_layouts: &[&BindGroupLayout],
        options: &PipelineOptions,
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
        };

        if !self.pipelines.contains_key(&key) {
            self.load_shader(shader_path);
            let pipeline = self.create_pipeline(&key, bind_group_layouts, options);
            self.pipelines.insert(key.clone(), pipeline);
        }

        self.pipelines.get(&key).unwrap()
    }

    /// Reload shaders from disk. Pipelines using reloaded shaders will be recreated on next use.
    pub fn reload_shaders(&mut self, paths: &[PathBuf]) {
        for path in paths {
            if self.shaders.contains_key(path) {
                self.load_shader(path);
            }
        }
        self.pipelines.retain(|key, _| !paths.contains(&key.shader_path));
    }

    /// Clear all cached pipelines and shaders.
    pub fn clear(&mut self) {
        self.shaders.clear();
        self.pipelines.clear();
    }

    fn load_shader(&mut self, path: &Path) {
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to load shader {:?}: {}", path, e));

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: path.to_str(),
            source: ShaderSource::Wgsl(source.into()),
        });

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
