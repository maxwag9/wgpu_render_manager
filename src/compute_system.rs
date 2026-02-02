// compute_system.rs
#![allow(dead_code)]
use std::collections::HashMap;
use std::fs;
use wgpu::*;

/// Options for compute dispatch
pub struct ComputePipelineOptions {
    pub dispatch_size: [u32; 3],
}

impl Default for ComputePipelineOptions {
    fn default() -> Self {
        Self {
            dispatch_size: [1, 1, 1],
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    shader_path: String,
    input_specs: Vec<(TextureFormat, u32, bool)>, // (format, sample_count, is_filterable)
    output_formats: Vec<TextureFormat>,
    uniform_count: usize,
}

struct CachedPipeline {
    pipeline: ComputePipeline,
    bind_group_layouts: [BindGroupLayout; 3],
}

/// High-level compute shader runner with automatic pipeline caching.
///
/// `ComputeSystem` provides a structured way to execute compute shaders
/// with multiple input textures, output storage textures, and optional
/// uniform buffers, while reusing pipelines whenever possible.
///
/// ## Features
/// - Automatic compute pipeline caching
/// - Flexible input and output texture bindings
/// - Optional uniform buffer bindings
/// - Built-in sampler for input textures
///
/// ## Design notes
/// - Pipelines are cached based on shader path, texture formats,
///   MSAA sample counts, and uniform count
/// - Bind group layouts are generated dynamically per pipeline
/// - This type owns its own command encoder per dispatch
///
/// This is intended for procedural texture generation, post-processing,
/// and general GPU compute workloads.
pub struct ComputeSystem {
    device: Device,
    queue: Queue,
    pipeline_cache: HashMap<PipelineKey, CachedPipeline>,
    filtering_sampler: Sampler,
    non_filtering_sampler: Sampler,
}

impl ComputeSystem {
    /// Create a new `ComputeSystem`.
    ///
    /// The provided `device` and `queue` are cloned internally (cheap, just handles).
    /// A default linear sampler is created and reused for all input textures.
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let device = device.clone();
        let queue = queue.clone();

        let filtering_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("compute_filtering_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        let non_filtering_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("compute_non_filtering_sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            device,
            queue,
            pipeline_cache: HashMap::new(),
            filtering_sampler,
            non_filtering_sampler,
        }
    }

    /// Execute a compute shader.
    ///
    /// This method creates (or reuses) a cached compute pipeline, sets up
    /// bind groups, dispatches workgroups, and submits the command buffer.
    ///
    /// ## Bind group layout
    /// - `@group(0)`: input textures + sampler
    ///   - `binding 0..n`: input texture views
    ///   - `binding n`: shared sampler
    /// - `@group(1)`: output storage textures
    ///   - `binding 0..m`: output texture views
    /// - `@group(2)`: uniform buffers
    ///   - `binding 0..k`: uniform buffers
    ///
    /// Empty input, output, or uniform lists result in empty bind groups
    /// for the corresponding slots.
    ///
    /// ## Texture handling
    /// - MSAA input textures are detected automatically via `sample_count`
    /// - Depth textures use depth sampling where applicable
    ///
    /// ## Parameters
    /// - `label`: Debug label for the command encoder and compute pass
    /// - `input_views`: Read-only input textures
    /// - `output_views`: Write-only storage textures
    /// - `shader_path`: Path to the WGSL compute shader
    /// - `options`: Compute pipeline and dispatch configuration
    /// - `uniforms`: Optional uniform buffers
    ///
    /// ## Notes
    /// - The shader entry point must be named `main`
    /// - The dispatch size is taken from `options.dispatch_size`
    ///
    /// ## WGSL expectations
    /// - Entry point: `@compute @workgroup_size(...) fn main()`
    /// - Input textures must match the order provided to `compute`
    /// - Output textures must be declared as `texture_storage_2d`
    /// - Uniform buffers must match binding indices exactly
    pub fn compute(
        &mut self,
        label: &str,
        input_views: Vec<&TextureView>,
        output_views: Vec<&TextureView>,
        shader_path: &str,
        options: ComputePipelineOptions,
        uniforms: &[&Buffer],
    ) {
        let input_specs: Vec<_> = input_views
            .iter()
            .map(|v| {
                let tex = v.texture();
                let format = tex.format();
                let sample_count = tex.sample_count();
                let is_filterable = self.is_format_filterable(format, sample_count);
                (format, sample_count, is_filterable)
            })
            .collect();

        let output_formats: Vec<_> = output_views.iter().map(|v| v.texture().format()).collect();

        let key = PipelineKey {
            shader_path: shader_path.to_string(),
            input_specs: input_specs.clone(),
            output_formats: output_formats.clone(),
            uniform_count: uniforms.len(),
        };

        // Get or create cached pipeline
        if !self.pipeline_cache.contains_key(&key) {
            let cached =
                self.create_pipeline(shader_path, &input_specs, &output_formats, uniforms.len());
            self.pipeline_cache.insert(key.clone(), cached);
        }
        let cached = self.pipeline_cache.get(&key).unwrap();

        // Determine if we can use filtering sampler (all non-depth, non-msaa inputs must be filterable)
        let use_filtering = input_specs.iter().all(|(format, sample_count, is_filterable)| {
            format.has_depth_aspect() || *sample_count > 1 || *is_filterable
        });

        // Create bind groups
        let input_bg = self.create_input_bind_group(&cached.bind_group_layouts[0], &input_views, use_filtering);
        let output_bg = self.create_output_bind_group(&cached.bind_group_layouts[1], &output_views);
        let uniform_bg = self.create_uniform_bind_group(&cached.bind_group_layouts[2], uniforms);

        // Execute
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &input_bg, &[]);
            pass.set_bind_group(1, &output_bg, &[]);
            pass.set_bind_group(2, &uniform_bg, &[]);
            pass.dispatch_workgroups(
                options.dispatch_size[0],
                options.dispatch_size[1],
                options.dispatch_size[2],
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    // let mut compute = ComputeSystem::new(&device, &queue); // Caches device and queue so you don't have to pass it in again
    // compute.compute(
    //     "Resolve Depth",             // Label
    //     vec![&msaa_depth_view],      // Input Views
    //     vec![&resoloved_depth_view], // Input Views
    //     shader_path,                 // Shader Path
    //     options,                     // ComputePipelineOptions
    //     &[&camera_buffer]            // Optional Buffers
    // );
    // let mut render_manager = RenderManager::new(&device, &queue, texture_shaders_path);
    // render_manager.render(
    //     &[TextureKey],               // TextureKeys for included procedural textures
    //     shader_path,                 // Shader Path
    //     options,                     // PipelineOptions
    //     &[&camera_buffer],           // Optional Buffers
    //     pass                         // Your own pass!
    // );

    fn create_pipeline(
        &self,
        shader_path: &str,
        input_specs: &[(TextureFormat, u32, bool)], // (format, sample_count, is_filterable)
        output_formats: &[TextureFormat],
        uniform_count: usize,
    ) -> CachedPipeline {
        let shader_source = fs::read_to_string(shader_path)
            .unwrap_or_else(|_| panic!("Failed to read shader: {}", shader_path));

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_path),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Determine if we can use filtering sampler
        let use_filtering = input_specs.iter().all(|(format, sample_count, is_filterable)| {
            format.has_depth_aspect() || *sample_count > 1 || *is_filterable
        });

        // Group 0: inputs + sampler
        let mut input_entries: Vec<BindGroupLayoutEntry> = input_specs
            .iter()
            .enumerate()
            .map(|(i, (format, sample_count, is_filterable))| {
                let multisampled = *sample_count > 1;
                let sample_type = if format.has_depth_aspect() {
                    TextureSampleType::Depth
                } else {
                    TextureSampleType::Float {
                        filterable: *is_filterable && !multisampled,
                    }
                };

                BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type,
                        view_dimension: TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                }
            })
            .collect();

        if !input_specs.is_empty() {
            let sampler_type = if use_filtering {
                SamplerBindingType::Filtering
            } else {
                SamplerBindingType::NonFiltering
            };

            input_entries.push(BindGroupLayoutEntry {
                binding: input_specs.len() as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(sampler_type),
                count: None,
            });
        }

        let input_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute_input_layout"),
                entries: &input_entries,
            });

        // Group 1: outputs
        let output_entries: Vec<BindGroupLayoutEntry> = output_formats
            .iter()
            .enumerate()
            .map(|(i, format)| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: *format,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            })
            .collect();

        let output_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute_output_layout"),
                entries: &output_entries,
            });

        // Group 2: uniforms
        let uniform_entries: Vec<BindGroupLayoutEntry> = (0..uniform_count)
            .map(|i| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let uniform_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute_uniform_layout"),
                entries: &uniform_entries,
            });

        let bind_group_layouts = [input_layout, output_layout, uniform_layout];

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(),
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(shader_path),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        CachedPipeline {
            pipeline,
            bind_group_layouts,
        }
    }

    fn create_input_bind_group(
        &self,
        layout: &BindGroupLayout,
        views: &[&TextureView],
        use_filtering: bool,
    ) -> BindGroup {
        let mut entries: Vec<BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, view)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::TextureView(view),
            })
            .collect();

        if !views.is_empty() {
            let sampler = if use_filtering {
                &self.filtering_sampler
            } else {
                &self.non_filtering_sampler
            };

            entries.push(BindGroupEntry {
                binding: views.len() as u32,
                resource: BindingResource::Sampler(sampler),
            });
        }

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_input_bg"),
            layout,
            entries: &entries,
        })
    }

    fn create_output_bind_group(
        &self,
        layout: &BindGroupLayout,
        views: &[&TextureView],
    ) -> BindGroup {
        let entries: Vec<BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, view)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::TextureView(view),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_output_bg"),
            layout,
            entries: &entries,
        })
    }

    fn create_uniform_bind_group(
        &self,
        layout: &BindGroupLayout,
        uniforms: &[&Buffer],
    ) -> BindGroup {
        let entries: Vec<BindGroupEntry> = uniforms
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_uniform_bg"),
            layout,
            entries: &entries,
        })
    }

    /// Clear the internal compute pipeline cache.
    ///
    /// Call this if compute shaders on disk have changed and pipelines
    /// need to be recreated.
    pub fn invalidate_cache(&mut self) {
        self.pipeline_cache.clear();
    }
    fn is_format_filterable(&self, format: TextureFormat, sample_count: u32) -> bool {
        if sample_count > 1 {
            // MSAA textures use textureLoad, filterability doesn't apply
            return false;
        }

        // Determine the appropriate aspect for the format
        let aspect = figure_out_aspect(format);

        // Depth/stencil textures have their own sample type, not filterable float
        if format.has_depth_aspect() || format.has_stencil_aspect() {
            return false;
        }

        // Check if the format supports filtering with current device features
        match format.sample_type(aspect, Some(self.device.features())) {
            Some(TextureSampleType::Float { filterable }) => filterable,
            _ => false,
        }
    }
}
pub(crate) fn figure_out_aspect(format: TextureFormat) -> Option<TextureAspect> {
    if format.has_depth_aspect() && format.has_stencil_aspect() {
        panic!("Fullscreen Debug Render received a Depth texture with both Depth aspect and Stencil aspect, which wgpu doesn't allow to be used in shaders together. \n \n \
        Solution: Make another view of the same texture with TextureAspect::DepthOnly and pass that in instead. \n")
    } else if format.has_depth_aspect() {
        Some(TextureAspect::DepthOnly)
    } else if format.has_stencil_aspect() {
        Some(TextureAspect::StencilOnly)
    } else {
        None
    }
}