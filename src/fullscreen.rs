// fullscreen.rs
#![allow(dead_code)]
use std::collections::HashMap;
use wgpu::*;
use wgpu::util::DeviceExt;
use crate::compute_system::figure_out_aspect;

const FULLSCREEN_COLOR_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0) var s_tex: sampler;
@group(0) @binding(1) var t_tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_tex, s_tex, in.uv);
}
"#;

const FULLSCREEN_DEPTH_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct DepthParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
};
@group(0) @binding(0) var s_depth: sampler;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(1) @binding(0) var<uniform> params: DepthParams;

fn linearize_depth(d: f32, near: f32, far: f32, reversed: bool) -> f32 {
    if reversed {
        return near / d;
    } else {
        return (near * far) / (far - d * (far - near));
    }
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureSample(t_depth, s_depth, in.uv);
    let z = linearize_depth(d, params.near, params.far, params.reversed_z != 0u);
    let v01 = 1.0 - clamp(z / params.far, 0.0, 1.0);
    let v = pow(v01, params.power);
    return vec4<f32>(v, v, v, 1.0);
}
"#;
const FULLSCREEN_COLOR_MSAA_SHADER: &str = r#"
struct DepthParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}
@group(0) @binding(0) var t_tex: texture_multisampled_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(t_tex);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));
    var c = vec4<f32>(0.0);
    for (var i = 0u; i < textureNumSamples(t_tex); i++) {
        c += textureLoad(t_tex, coord, i);
    }
    return c / f32(textureNumSamples(t_tex));
}
"#;

const FULLSCREEN_DEPTH_MSAA_SHADER: &str = r#"
struct DepthParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}
fn linearize_depth(d: f32, near: f32, far: f32, reversed: bool) -> f32 {
    if reversed {
        return near / d;
    } else {
        return (near * far) / (far - d * (far - near));
    }
}
@group(0) @binding(0) var t_depth: texture_depth_multisampled_2d;
@group(1) @binding(0) var<uniform> params: DepthParams;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(t_depth);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));

    var d = 0.0;
    for (var i = 0u; i < textureNumSamples(t_depth); i++) {
        d += textureLoad(t_depth, coord, i);
    }
    d /= f32(textureNumSamples(t_depth));

    let z = linearize_depth(d, params.near, params.far, params.reversed_z != 0u);
    let v01 = 1.0 - clamp(z / params.far, 0.0, 1.0);
    let v = pow(v01, params.power);
    return vec4<f32>(v, v, v, 1.0);
}
"#;
const FULLSCREEN_RED_TO_GRAYSCALE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0) var s_tex: sampler;
@group(0) @binding(1) var t_tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(t_tex, s_tex, in.uv);
    let v = c.r;
    return vec4<f32>(v, v, v, 1.0);
}
"#;

const FULLSCREEN_RED_TO_GRAYSCALE_MSAA_SHADER: &str = r#"
struct DepthParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}
@group(0) @binding(0) var t_tex: texture_multisampled_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(t_tex);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));
    var c = vec4<f32>(0.0);
    for (var i = 0u; i < textureNumSamples(t_tex); i++) {
        c += textureLoad(t_tex, coord, i);
    }

    let v = c.r;
    return vec4<f32>(v, v, v, 1.0);
}
"#;
const FULLSCREEN_COLOR_UNFILTERABLE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0) var t_tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(t_tex);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));
    return textureLoad(t_tex, coord, 0);
}
"#;

const FULLSCREEN_RED_TO_GRAYSCALE_UNFILTERABLE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0) var t_tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(t_tex);
    let coord = vec2<i32>(in.uv * vec2<f32>(dims));
    let c = textureLoad(t_tex, coord, 0);
    let v = c.r;
    return vec4<f32>(v, v, v, 1.0);
}
"#;
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DepthDebugParams {
    pub near: f32,
    pub far: f32,
    pub power: f32,        // e.g. 20.0
    pub reversed_z: u32,   // 0 or 1
    pub msaa_samples: u32, // 1,2,4,8...
}

/// Type of debug visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugVisualization {
    Color,
    RedToGrayscale,
    Depth,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PipelineKind {
    Color,
    RedToGrayscale,
    Depth,
}

impl PipelineKind {
    fn from_debug_visualization(visualization: DebugVisualization) -> Self {
        match visualization {
            DebugVisualization::Color => Self::Color,
            DebugVisualization::RedToGrayscale => Self::RedToGrayscale,
            DebugVisualization::Depth => Self::Depth
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    kind: PipelineKind,
    target_format: TextureFormat,
    source_is_msaa: bool,
    source_is_filterable: bool,
    target_sample_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BindGroupKey {
    view_ptr: usize,
    kind: PipelineKind,
    is_filterable: bool,
}

/// Renders textures to fullscreen quads for debugging and visualization.
pub struct FullscreenRenderer {
    device: Device,
    queue: Queue,

    color_shader: ShaderModule,
    color_unfilterable_shader: ShaderModule,
    color_msaa_shader: ShaderModule,
    red_to_grayscale_shader: ShaderModule,
    red_to_grayscale_unfilterable_shader: ShaderModule,
    red_to_grayscale_msaa_shader: ShaderModule,
    depth_shader: ShaderModule,
    depth_msaa_shader: ShaderModule,

    color_bgl: BindGroupLayout,
    color_unfilterable_bgl: BindGroupLayout,
    color_msaa_bgl: BindGroupLayout,
    depth_bgl: BindGroupLayout,
    depth_msaa_bgl: BindGroupLayout,
    depth_params_bgl: BindGroupLayout,

    linear_sampler: Sampler,
    nearest_sampler: Sampler,

    pipelines: HashMap<PipelineKey, RenderPipeline>,
    bind_groups: HashMap<BindGroupKey, BindGroup>,

    depth_params_buffer: Option<Buffer>,
    depth_params_bind_group: Option<BindGroup>,
}

impl FullscreenRenderer {
    pub fn new(device: Device, queue: Queue) -> Self {
        let depth_params_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("depth params bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let color_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen color shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_COLOR_SHADER.into()),
        });
        let color_unfilterable_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen color unfilterable shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_COLOR_UNFILTERABLE_SHADER.into()),
        });
        let color_msaa_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen color MSAA shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_COLOR_MSAA_SHADER.into()),
        });
        let red_to_grayscale_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen grayscale shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_RED_TO_GRAYSCALE_SHADER.into()),
        });
        let red_to_grayscale_unfilterable_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen grayscale unfilterable shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_RED_TO_GRAYSCALE_UNFILTERABLE_SHADER.into()),
        });
        let red_to_grayscale_msaa_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen grayscale shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_RED_TO_GRAYSCALE_MSAA_SHADER.into()),
        });
        let depth_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen depth shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_DEPTH_SHADER.into()),
        });
        let depth_msaa_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fullscreen depth MSAA shader"),
            source: ShaderSource::Wgsl(FULLSCREEN_DEPTH_MSAA_SHADER.into()),
        });
        let color_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fullscreen color bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let color_unfilterable_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fullscreen color unfilterable bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let color_msaa_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fullscreen color msaa bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });

        let depth_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fullscreen depth bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let depth_msaa_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fullscreen depth msaa bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });

        let linear_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Fullscreen Linear Sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Fullscreen Nearest Sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            device,
            queue,
            color_shader,
            color_unfilterable_shader,
            color_msaa_shader,
            red_to_grayscale_shader,
            red_to_grayscale_unfilterable_shader,
            red_to_grayscale_msaa_shader,
            depth_shader,
            depth_msaa_shader,
            color_bgl,
            color_unfilterable_bgl,
            color_msaa_bgl,
            depth_bgl,
            depth_msaa_bgl,
            depth_params_bgl,
            linear_sampler,
            nearest_sampler,
            pipelines: HashMap::new(),
            bind_groups: HashMap::new(),
            depth_params_buffer: None,
            depth_params_bind_group: None,
        }
    }

    /// Renders a texture to the current render pass as a fullscreen quad.
    ///
    /// This function is intended for **debugging and visualization** of GPU
    /// textures such as color buffers, depth buffers, or intermediate render
    /// targets.
    ///
    /// ## What this does
    /// - Selects or creates a render pipeline based on [`DebugVisualization`]
    /// - Binds the provided texture as `@group(0)`
    /// - Optionally binds depth visualization parameters at `@group(1)`
    /// - Draws a fullscreen quad using a single draw call and internal shaders
    ///
    /// ## Target View
    /// `target_view` **must match** the color attachment of the render pass.
    /// It is used to infer the render target format and MSAA sample count
    /// for pipeline selection.
    ///
    /// ## Parameters
    /// - `texture`: Texture view to visualize
    /// - `visualization`: What the texture should be interpreted as (color, depth, etc.)
    /// - `target_format`: Format of the render target this pass is writing to
    /// - `pass`: Active render pass to record commands into
    ///
    /// ### Panics
    /// This function may panic if internal pipeline or bind group creation fails.
    ///
    /// ### Errors
    /// wgpu validation errors may occur if the texture, target view,
    /// or render pass configuration are incompatible.
    ///
    /// ## Example
    /// ```no_run
    /// // Inside a render pass
    /// fullscreen_renderer.render(
    ///     &color_view,
    ///     DebugVisualization::Color,
    ///     &target_view,
    ///     &mut render_pass,
    /// );
    /// ```
    pub fn render(
        &mut self,
        texture: &TextureView,
        visualization_type: DebugVisualization,
        target_view: &TextureView,
        pass: &mut RenderPass,
    ) {
        //texture.texture().
        let binding_type = infer_texture_binding_type(texture, &self.device);
        let sample_type = match binding_type {
            BindingType::Texture { sample_type, .. } => sample_type,
            _ => unreachable!("infer_texture_binding_type always returns BindingType::Texture"),
        };

        let kind = if matches!(sample_type, TextureSampleType::Depth) {
            PipelineKind::Depth
        } else {
            PipelineKind::from_debug_visualization(visualization_type)
        };

        let is_filterable = matches!(sample_type, TextureSampleType::Float { filterable: true });

        let pipeline_msaa_samples = target_view.texture().sample_count();
        let target_format = target_view.texture().format();

        let pipeline = self.get_or_create_pipeline(texture, kind, target_format, pipeline_msaa_samples, is_filterable);
        pass.set_pipeline(pipeline);

        let bind_group = self.get_or_create_bind_group(texture, kind, is_filterable);
        pass.set_bind_group(0, bind_group, &[]);

        if kind == PipelineKind::Depth {
            if let Some(bg) = &self.depth_params_bind_group {
                pass.set_bind_group(1, bg, &[]);
            }
        }

        pass.draw(0..4, 0..1);
    }

    /// Clear cached bind groups (call when textures are recreated).
    pub(crate) fn invalidate_bind_groups(&mut self) {
        self.bind_groups.clear();
    }

    pub(crate) fn update_depth_params(&mut self, params: DepthDebugParams) {
        if let Some(buf) = &self.depth_params_buffer {
            self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&params));
        } else {
            let buf = self.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("depth params buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

            let bg = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("depth params bind group"),
                layout: &self.depth_params_bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });

            self.depth_params_buffer = Some(buf);
            self.depth_params_bind_group = Some(bg);
        }
    }

    fn get_or_create_pipeline(
        &mut self,
        texture: &TextureView,
        kind: PipelineKind,
        target_format: TextureFormat,
        target_sample_count: u32,
        is_filterable: bool,
    ) -> &RenderPipeline {
        let source_is_msaa = texture.texture().sample_count() > 1;
        let key = PipelineKey {
            kind,
            target_format,
            source_is_msaa,
            source_is_filterable: is_filterable,
            target_sample_count,
        };

        if !self.pipelines.contains_key(&key) {
            let pipeline = self.create_pipeline(kind, target_format, target_sample_count, source_is_msaa, is_filterable);
            self.pipelines.insert(key, pipeline);
        }

        self.pipelines.get(&key).unwrap()
    }

    fn create_pipeline(
        &self,
        kind: PipelineKind,
        target_format: TextureFormat,
        pipeline_msaa_samples: u32,
        source_is_msaa: bool,
        is_filterable: bool,
    ) -> RenderPipeline {
        let (shader, bgl) = match (kind, source_is_msaa, is_filterable) {
            // MSAA sources always use MSAA shaders (they use textureLoad)
            (PipelineKind::Color, true, _) => (&self.color_msaa_shader, &self.color_msaa_bgl),
            (PipelineKind::RedToGrayscale, true, _) => (&self.red_to_grayscale_msaa_shader, &self.color_msaa_bgl),
            (PipelineKind::Depth, true, _) => (&self.depth_msaa_shader, &self.depth_msaa_bgl),

            // Non-MSAA filterable float textures use sampler-based shaders
            (PipelineKind::Color, false, true) => (&self.color_shader, &self.color_bgl),
            (PipelineKind::RedToGrayscale, false, true) => (&self.red_to_grayscale_shader, &self.color_bgl),

            // Non-MSAA non-filterable float textures use textureLoad shaders
            (PipelineKind::Color, false, false) => (&self.color_unfilterable_shader, &self.color_unfilterable_bgl),
            (PipelineKind::RedToGrayscale, false, false) => (&self.red_to_grayscale_unfilterable_shader, &self.color_unfilterable_bgl),

            // Depth textures (non-MSAA) use their own sampler-based shader
            (PipelineKind::Depth, false, _) => (&self.depth_shader, &self.depth_bgl),
        };

        let bind_group_layouts: Vec<&BindGroupLayout> = if kind == PipelineKind::Depth {
            vec![bgl, &self.depth_params_bgl]
        } else {
            vec![bgl]
        };

        let layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("fullscreen pipeline layout"),
            bind_group_layouts: &bind_group_layouts,
            immediate_size: 0,
        });

        self.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("fullscreen pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: pipeline_msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        })
    }

    fn get_or_create_bind_group(
        &mut self,
        view: &TextureView,
        kind: PipelineKind,
        is_filterable: bool,
    ) -> &BindGroup {
        let key = BindGroupKey {
            view_ptr: view as *const TextureView as usize,
            kind,
            is_filterable,
        };

        let is_msaa = view.texture().sample_count() > 1;

        if !self.bind_groups.contains_key(&key) {
            let bg = match (kind, is_msaa, is_filterable) {
                // MSAA textures: no sampler, just texture at binding 0
                (PipelineKind::Color | PipelineKind::RedToGrayscale, true, _) => {
                    self.device.create_bind_group(&BindGroupDescriptor {
                        label: Some("Fullscreen Color MSAA Bind Group"),
                        layout: &self.color_msaa_bgl,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(view),
                        }],
                    })
                }
                (PipelineKind::Depth, true, _) => {
                    self.device.create_bind_group(&BindGroupDescriptor {
                        label: Some("Fullscreen Depth MSAA Bind Group"),
                        layout: &self.depth_msaa_bgl,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(view),
                        }],
                    })
                }

                // Non-MSAA unfilterable: no sampler, just texture at binding 0
                (PipelineKind::Color | PipelineKind::RedToGrayscale, false, false) => {
                    self.device.create_bind_group(&BindGroupDescriptor {
                        label: Some("Fullscreen Color Unfilterable Bind Group"),
                        layout: &self.color_unfilterable_bgl,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(view),
                        }],
                    })
                }

                // Non-MSAA filterable: sampler + texture
                (PipelineKind::Color | PipelineKind::RedToGrayscale, false, true) => {
                    self.device.create_bind_group(&BindGroupDescriptor {
                        label: Some("Fullscreen Color Bind Group"),
                        layout: &self.color_bgl,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Sampler(&self.linear_sampler),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::TextureView(view),
                            },
                        ],
                    })
                }

                // Depth non-MSAA: nearest sampler + texture
                (PipelineKind::Depth, false, _) => {
                    self.device.create_bind_group(&BindGroupDescriptor {
                        label: Some("Fullscreen Depth Bind Group"),
                        layout: &self.depth_bgl,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Sampler(&self.nearest_sampler),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::TextureView(view),
                            },
                        ],
                    })
                }
            };

            self.bind_groups.insert(key, bg);
        }

        self.bind_groups.get(&key).unwrap()
    }
}

/// Infer a `BindingType::Texture` for a view + device.
/// Automatically determines the correct aspect for depth-stencil formats.
fn infer_texture_binding_type(view: &TextureView, device: &Device) -> BindingType {
    let format = view.texture().format();

    // Determine the appropriate aspect for sampling
    let aspect = figure_out_aspect(format);

    let sample_type = format
        .sample_type(aspect, Some(device.features()))
        .unwrap_or_else(|| {
            panic!(
                "unsupported texture format {:?} with aspect {:?} for sampling",
                format, aspect
            )
        });

    let multisampled = view.texture().sample_count() > 1;

    BindingType::Texture {
        sample_type,
        view_dimension: TextureViewDimension::D2,
        multisampled,
    }
}

/// Infer an appropriate sampler binding type from the texture sample type.
/// Note: depth textures are usually sampled with a comparison sampler and shader uses textureSampleCmp.
fn infer_sampler_binding_type(sample_type: TextureSampleType) -> SamplerBindingType {
    match sample_type {
        TextureSampleType::Depth => SamplerBindingType::Comparison,
        TextureSampleType::Float { filterable } => {
            if filterable { SamplerBindingType::Filtering } else { SamplerBindingType::NonFiltering }
        }
        TextureSampleType::Uint | TextureSampleType::Sint => SamplerBindingType::NonFiltering,
    }
}
