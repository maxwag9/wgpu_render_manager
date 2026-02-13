use std::collections::{HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, Device, Queue, RenderPass, TextureView};
use crate::bind_groups::{LayoutKey, MaterialBindGroups};
use crate::compute_system::{BufferSet, ComputePipelineOptions, ComputeSystem};
use crate::fullscreen::{DebugVisualization, DepthDebugParams, FullscreenRenderer};
use crate::generator::{TextureGenerator, TextureKey};
use crate::pipelines::{PipelineCache, PipelineOptions};

#[derive(Clone, Hash, PartialEq, Eq)]
struct UniformBindGroupKey(u64);

impl UniformBindGroupKey {
    fn from_buffers(buffers: &[&Buffer]) -> Self {
        let mut hasher = DefaultHasher::default();
        for buffer in buffers {
            buffer.hash(&mut hasher);
        }

        Self(hasher.finish())
    }
}

/// High-level render manager combining texture generation, pipeline caching,
/// material bind groups, and fullscreen debug rendering.
///
/// `RenderManager` is designed to remove most of the boilerplate involved in
/// setting up render pipelines and bind groups when working with `wgpu`,
/// while still keeping behavior explicit and predictable.
///
/// ## Responsibilities
/// - Procedural texture generation via compute shaders
/// - Render pipeline creation and caching
/// - Automatic material bind group creation
/// - Optional uniform bind group management
/// - Fullscreen debug visualization of textures
///
/// ## What this does *not* do
/// - It does not manage render passes or frame submission
/// - It does not hide `wgpu` concepts like bind groups or pipelines
/// - It does not perform draw calls for you (except fullscreen debug)
///
/// ## Design goals
/// - Minimal boilerplate for common rendering paths
/// - Explicit control over layouts and bindings
/// - Safe reuse of pipelines and bind groups
///
/// Most users will interact with this type as their primary entry point
/// into the crate.
pub struct RenderManager {
    device: Device,
    queue: Queue,
    generator: TextureGenerator,
    pipeline_cache: PipelineCache,
    fullscreen: FullscreenRenderer,
    materials: MaterialBindGroups,
    compute_system: ComputeSystem,
    uniform_bind_groups: HashMap<UniformBindGroupKey, BindGroup>,
    defines: HashMap<String, bool>,
}

impl RenderManager {
    /// Create a new `RenderManager`.
    ///
    /// Procedural texture compute shaders will be loaded from
    /// `texture_shader_dir`. The directory is watched logically, meaning
    /// shaders can later be reloaded without recreating the manager.
    ///
    /// The provided `device` and `queue` are cloned internally and reused
    /// by all sub-systems.
    /// Cloning `device` and `queue` is very cheap, as they are just handles in wgpu.
    pub fn new(device: &Device, queue: &Queue, texture_shader_dir: PathBuf) -> Self {
        let generator = TextureGenerator::new(device.clone(), queue.clone(), texture_shader_dir);
        let pipeline_cache = PipelineCache::new(device.clone());
        let fullscreen = FullscreenRenderer::new(device.clone(), queue.clone());
        let materials = MaterialBindGroups::new(device.clone());
        let compute_system = ComputeSystem::new(device, queue);
        Self {
            device: device.clone(),
            queue: queue.clone(),
            generator,
            pipeline_cache,
            fullscreen,
            materials,
            compute_system,
            uniform_bind_groups: HashMap::new(),
            defines: HashMap::new(),
        }
    }

    /// Returns a reference to the underlying `wgpu::Device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the underlying `wgpu::Queue`.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Access the procedural texture generator.
    ///
    /// This allows manual creation, inspection, or reuse of generated
    /// texture views outside the high-level render APIs.
    pub fn generator(&mut self) -> &mut TextureGenerator {
        &mut self.generator
    }

    /// Access the internal render pipeline cache.
    ///
    /// Useful for advanced use cases such as manual pipeline preloading
    /// or shader hot-reloading.
    pub fn pipeline_cache(&mut self) -> &mut PipelineCache {
        &mut self.pipeline_cache
    }

    /// Access the fullscreen debug renderer.
    ///
    /// This renderer is used internally by
    /// [`render_fullscreen_debug`](Self::render_fullscreen_debug),
    /// but can also be used directly for custom debug workflows.
    pub fn fullscreen(&mut self) -> &mut FullscreenRenderer {
        &mut self.fullscreen
    }

    /// Access the compute system.
    ///
    /// This renderer is used internally.
    pub fn compute_system(&mut self) -> &mut ComputeSystem {
        &mut self.compute_system
    }

    /// Render using procedurally generated textures.
    ///
    /// This method resolves textures using the internal
    /// [`TextureGenerator`] and sets up the render pipeline
    /// and bind groups automatically.
    ///
    /// Only pipeline and bind groups are set.
    /// You must call `pass.draw()` yourself.
    ///
    /// Uses [`render_with_textures()`](Self::render_with_textures) underneath.
    /// ## Texture loading
    /// Each [`TextureKey`] resolves to a compute shader where:
    /// - `shader_id` is lowercased
    /// - `.wgsl` is appended
    ///
    /// ## Parameters
    /// - `texture_keys`: Keys describing which textures to generate or reuse
    /// - `shader_path`: Path to the render shader
    /// - `options`: Pipeline configuration options
    /// - `uniforms`: Uniform buffers bound after material bindings
    /// - `pass`: Active render pass
    ///
    /// ## Shader Binding layout
    /// - `@group(0) @binding(0)`: trilinear sampler
    /// - `@group(0) @binding(0..n)`: textures as texture_2d<f32> (Rgba8Unorm)
    /// - `@group(0) @binding(n+1)`: (optional) shadow_sampler
    /// - `@group(0) @binding(n+2)`: (optional) shadow textures as texture_depth_2d_array
    /// - `@group(1) @binding(0..n)`: uniforms, in the same order as input
    ///
    /// WGSL shaders are compiled via [`compile_wgsl()`](crate::shader_preprocessing::compile_wgsl), which adds a small
    /// compile-time preprocessing layer (`#ifdef`, `#include`) on top of
    /// standard WGSL before passing it to wgpu.
    /// ## Example
    /// ```no_run
    /// // Inside a render pass
    /// render_manager.render_with_textures(
    ///     &texture_keys.as_slice(),  // Texture Keys
    ///     shader_path.as_path(),     // Shader Path
    ///     options,                   // Pipeline Options
    ///     &[&uniforms_buffer],       // Buffers
    ///     &mut render_pass,          // Render Pass
    /// );
    /// ```
    pub fn render(
        &mut self,
        texture_keys: &[TextureKey],
        shader_path: &Path,
        options: &PipelineOptions,
        uniforms: &[&Buffer],
        pass: &mut RenderPass,
    ) {
        // Cloning TextureView is cheap — it's just a handle to the underlying GPU object.
        let mut owned_views: Vec<TextureView> = Vec::with_capacity(texture_keys.len());
        for key in texture_keys {
            let v_ref = self.generator.get_or_create(key);
            owned_views.push(v_ref.clone());
        }

        let view_refs: Vec<&TextureView> = owned_views.iter().collect();

        self.render_with_textures(&view_refs, shader_path, options, uniforms, pass);
    }

    /// Render using pre-existing texture views.
    ///
    /// Use this when textures originate from sources other than the
    /// procedural texture generator, such as render targets or external
    /// textures.
    ///
    /// This method automatically:
    /// - Creates or reuses a material bind group
    /// - Sets up optional shadow bindings
    /// - Handles uniform bind groups if provided
    ///
    /// Like [`render`](Self::render), this does not issue a draw call.
    ///
    /// ## Shader Binding layout
    /// - `@group(0) @binding(0)`: trilinear sampler
    /// - `@group(0) @binding(0..n)`: textures as texture_2d<f32> or texture_multisampled_2d<f32>
    /// - `@group(0) @binding(n+1)`: (optional) shadow_sampler
    /// - `@group(0) @binding(n+2)`: (optional) shadow textures as texture_depth_2d_array
    /// - `@group(1) @binding(0..n)`: uniforms, in the same order as input
    ///
    /// WGSL shaders are compiled via [`compile_wgsl()`](crate::shader_preprocessing::compile_wgsl), which adds a small
    /// compile-time preprocessing layer (`#ifdef`, `#include`) on top of
    /// standard WGSL before passing it to wgpu.
    /// ## Example
    /// ```no_run
    /// // Inside a render pass
    /// render_manager.render_with_textures(
    ///     &texture_views.as_slice(), // Texture Views
    ///     shader_path.as_path(),     // Shader Path
    ///     options,                   // Pipeline Options
    ///     &[&uniforms_buffer],       // Buffers
    ///     &mut render_pass,          // Render Pass
    /// );
    /// ```

    pub fn render_with_textures(
        &mut self,
        texture_views: &[&TextureView],
        shader_path: &Path,
        options: &PipelineOptions,
        uniforms: &[&Buffer],
        pass: &mut RenderPass,
    ) {
        // Shadow pulled explicitly from pipeline options
        let shadow = options.shadow.as_ref().map(|s| (&s.sampler, &s.view));
        let has_shadow = shadow.is_some();

        // Ensure material layout exists and clone handle
        let _ = self.materials.layout(texture_views, has_shadow);
        let material_layout_handle = self
            .materials
            .layouts
            .get(&LayoutKey::from_views(texture_views, has_shadow))
            .expect("material layout must exist")
            .clone();

        // Uniform layout
        let uniform_count = uniforms.len();
        let mut owned_bgls: Vec<BindGroupLayout> =
            Vec::with_capacity(if uniform_count > 0 { 2 } else { 1 });

        owned_bgls.push(material_layout_handle);

        if uniform_count > 0 {
            let _ = self.pipeline_cache.uniform_layout(uniform_count);
            let uniform_layout_handle = self
                .pipeline_cache
                .uniform_layouts
                .get(&uniform_count)
                .expect("uniform layout must exist")
                .clone();
            owned_bgls.push(uniform_layout_handle);
        }

        // Local references only
        let bind_group_layout_refs: Vec<&BindGroupLayout> = owned_bgls.iter().collect();

        // Pipeline
        let pipeline_ref = self
            .pipeline_cache
            .get_or_create(shader_path, &bind_group_layout_refs, options, &self.defines);
        let pipeline = pipeline_ref.clone();
        pass.set_pipeline(&pipeline);

        // Material bind group
        let material_bg = self.materials.get_or_create(texture_views, shadow);
        pass.set_bind_group(0, material_bg, &[]);

        // Uniform bind group
        if uniform_count > 0 {
            let uniform_bg = self.get_or_create_uniform_bind_group(uniforms);
            pass.set_bind_group(1, uniform_bg, &[]);
        }
    }


    /// Render using fully custom bind group layouts and bind groups.
    ///
    /// This is an advanced API intended for cases where automatic
    /// material or uniform handling is insufficient.
    ///
    /// No assumptions are made about binding order or layout structure.
    /// Bind groups are bound sequentially starting at index 0.
    ///
    /// WGSL shaders are compiled via [`compile_wgsl()`](crate::shader_preprocessing::compile_wgsl), which adds a small
    /// compile-time preprocessing layer (`#ifdef`, `#include`) on top of
    /// standard WGSL before passing it to wgpu.
    pub fn render_with_layouts(
        &mut self,
        shader_path: &Path,
        bind_group_layouts: &[&BindGroupLayout],
        bind_groups: &[&BindGroup],
        options: &PipelineOptions,
        pass: &mut RenderPass,
    ) {
        let pipeline = self.pipeline_cache.get_or_create(shader_path, bind_group_layouts, options, &self.defines);
        pass.set_pipeline(pipeline);

        for (i, bg) in bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *bg, &[]);
        }
    }

    /// Render a fullscreen debug visualization of a texture.
    ///
    /// This is primarily intended for inspecting intermediate render
    /// targets, depth buffers, or compute outputs.
    ///
    /// ## Depth
    /// For depth texture visualizations, it is highly recommended to update the depth params using [`update_depth_params()`](Self::update_depth_params).
    ///
    pub fn render_fullscreen_debug(
        &mut self,
        texture: &TextureView,
        visualization_type: DebugVisualization,
        target_view: &TextureView,
        pass: &mut RenderPass,
    ) {
        self.fullscreen.render(texture, visualization_type, target_view, pass);
    }

    /// Execute a compute shader, optionally using an existing command encoder.
    ///
    /// This method creates (or reuses) a cached compute pipeline, sets up bind groups,
    /// and dispatches workgroups.
    ///
    /// If `encoder` is `Some(&mut encoder)`, the compute pass is recorded into the provided
    /// encoder (no finish/submit is performed).
    ///
    /// If `encoder` is `None`, a new command encoder is created, used for the pass,
    /// finished, and immediately submitted to the queue.
    ///
    /// ### SYNCHRONIZATION WARNING
    /// Creating a new encoder (i.e. passing `None`) while other command encoders are still
    /// open is **extremely dangerous**. It can cause GPU timeline desynchronization,
    /// resource hazards, validation layer errors, crashes, or silent corruption if the
    /// operations touch overlapping resources without explicit barriers.
    ///
    /// **ALWAYS prefer passing an existing encoder** when you're doing multiple
    /// compute/render operations in the same frame. Only use `None` for isolated,
    /// one-off dispatches.
    ///
    /// In debug builds, a loud runtime warning will be printed if you create a new
    /// encoder while others are active.
    ///
    /// ## Bind group layout
    /// - `@group(0)`: input textures + sampler
    ///   - `binding 0..n`: input texture views
    ///   - `binding n`: shared sampler
    /// - `@group(1)`: output storage textures
    ///   - `binding 0..m`: output texture views
    /// - `@group(2)`: uniform/storage(read/read_write) buffers
    ///   - `binding 0..k`: uniform/storage(read/read_write) buffers
    ///
    /// Empty input/output/uniform/storage(read/read_write) lists create empty bind groups for those slots.
    ///
    /// ## Texture handling
    /// - MSAA input textures are auto-detected via `sample_count`
    /// - Depth textures use depth sampling where applicable
    ///
    /// ## Parameters
    /// - `encoder`: Optional existing encoder to record into
    /// - `label`: Debug label for the encoder (if created) and compute pass
    /// - `input_views`: Read-only input texture views
    /// - `output_views`: Write-only storage texture views
    /// - `shader_path`: Path to the WGSL compute shader
    /// - `options`: Compute pipeline and dispatch configuration
    /// - `buffer_sets`: Optional uniform/storage(read/read_write) buffers
    ///
    /// ## Notes
    /// - Shader entry point must be `main`
    /// - Dispatch size comes from `options.dispatch_size`
    ///
    /// ## WGSL expectations
    /// - Entry point: `@compute @workgroup_size(...) fn main()`
    /// - Input textures must match the order given
    /// - Output textures must be `texture_storage_2d<... , write>`
    /// - Uniforms/Storage(read_write) must match binding indices exactly
    ///
    /// WGSL shaders are compiled via [`compile_wgsl()`](crate::shader_preprocessing::compile_wgsl), which adds a small
    /// compile-time preprocessing layer (`#ifdef`, `#include`) on top of
    /// standard WGSL before passing it to wgpu.
    pub fn compute(
        &mut self,
        encoder: Option<&mut CommandEncoder>,
        label: &str,
        input_views: Vec<&TextureView>,
        output_views: Vec<&TextureView>,
        shader_path: &PathBuf,
        options: ComputePipelineOptions,
        buffer_sets: &[BufferSet],
    ) {
        self.compute_system.compute(encoder, label, input_views, output_views, shader_path, options, buffer_sets, &self.defines);
    }

    /// Enables or disables a compile-time shader define.
    ///
    /// This updates the internal set of shader `defines` used during WGSL
    /// compilation. Defines control `#ifdef` / `#ifndef` blocks in shaders
    /// and are evaluated **at shader compile time**, not at runtime.
    ///
    /// When a define is enabled:
    /// - Shaders compiled afterward may select different code paths
    /// - Binding layouts may change
    /// - A new pipeline variant may be created
    ///
    /// When a define is disabled:
    /// - The corresponding `#ifdef` blocks are excluded
    ///
    /// ### Important
    ///
    /// - Changing a define does **not** retroactively affect already-created
    ///   pipelines or shader modules.
    /// - Call this **before** compiling or requesting a pipeline that depends
    ///   on the define.
    /// - This is typically used for feature toggles such as MSAA, shadows,
    ///   fog variants, or debug paths.
    ///
    /// ### Example
    ///
    /// ```text
    /// update_define("MSAA".into(), true);
    /// update_define("SSAO".into(), false);
    /// ```
    /// ## Used for:
    /// WGSL shaders are compiled via [`compile_wgsl()`](crate::shader_preprocessing::compile_wgsl), which adds a small
    /// compile-time preprocessing layer (`#ifdef`, `#include`) on top of
    /// standard WGSL before passing it to wgpu.
    pub fn update_define(&mut self, define: String, enabled: bool) { self.defines.insert(define, enabled); }
    /// Update parameters used for depth texture visualization.
    ///
    /// These parameters affect subsequent calls to
    /// [`render_fullscreen_debug`](Self::render_fullscreen_debug).
    pub fn update_depth_params(&mut self, params: DepthDebugParams) {
        self.fullscreen.update_depth_params(params);
    }

    /// Clear cached material and uniform bind groups.
    ///
    /// Call this after window resize, swapchain recreation,
    /// or when underlying textures are replaced.
    pub fn invalidate_bind_groups(&mut self) {
        self.materials.clear();
        self.fullscreen.invalidate_bind_groups();
        self.uniform_bind_groups.clear();
    }

    /// Reload render shaders from disk.
    ///
    /// Existing pipelines using these shaders will be recreated
    /// on next use.
    ///
    /// Useful for shader hot-reloading
    pub fn reload_render_shaders(&mut self, paths: &[PathBuf]) {
        self.pipeline_cache.reload_shaders(paths, &self.defines);
    }

    /// Reload procedural texture shaders and clear the texture cache.
    pub fn reload_texture_shaders(&mut self) {
        self.generator.reload_shaders();
    }

    /// Clear all internal caches.
    ///
    /// This includes pipelines, generated textures, and bind groups.
    pub fn clear_all(&mut self) {
        self.pipeline_cache.clear();
        self.generator.clear_cache();
        self.invalidate_bind_groups();
    }

    fn get_or_create_uniform_bind_group(&mut self, uniforms: &[&Buffer]) -> &BindGroup {
        let key = UniformBindGroupKey::from_buffers(uniforms);

        if !self.uniform_bind_groups.contains_key(&key) {
            let bg = self.pipeline_cache.create_uniform_bind_group(uniforms, "uniform bind group");
            self.uniform_bind_groups.insert(key.clone(), bg);
        }

        self.uniform_bind_groups.get(&key).unwrap()
    }
}
