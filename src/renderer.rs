use std::collections::HashMap;
use std::path::{Path, PathBuf};
use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, Queue, RenderPass, TextureFormat, TextureView};
use crate::bind_groups::MaterialBindGroups;
use crate::fullscreen::{DebugVisualization, DepthDebugParams, FullscreenRenderer};
use crate::generator::{TextureGenerator, TextureKey};
use crate::pipelines::{PipelineCache, PipelineOptions};

#[derive(Clone, Hash, PartialEq, Eq)]
struct UniformBindGroupKey(Vec<usize>);

impl UniformBindGroupKey {
    fn from_buffers(buffers: &[&Buffer]) -> Self {
        Self(buffers.iter().map(|b| *b as *const Buffer as usize).collect())
    }
}

/// High-level render manager combining texture generation, pipeline caching,
/// and material management.
pub struct RenderManager {
    device: Device,
    queue: Queue,
    generator: TextureGenerator,
    pipeline_cache: PipelineCache,
    fullscreen: FullscreenRenderer,
    materials: MaterialBindGroups,
    uniform_bind_groups: HashMap<UniformBindGroupKey, BindGroup>,
}

impl RenderManager {
    /// Create a new render manager.
    ///
    /// Procedural texture shaders will be loaded from `texture_shader_dir`.
    pub fn new(device: &Device, queue: &Queue, texture_shader_dir: PathBuf) -> Self {
        let generator = TextureGenerator::new(device.clone(), queue.clone(), texture_shader_dir);
        let pipeline_cache = PipelineCache::new(device.clone());
        let fullscreen = FullscreenRenderer::new(device.clone(), queue.clone());
        let materials = MaterialBindGroups::new(device.clone());

        Self {
            device: device.clone(),
            queue: queue.clone(),
            generator,
            pipeline_cache,
            fullscreen,
            materials,
            uniform_bind_groups: HashMap::new(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Access the texture generator directly.
    pub fn generator(&mut self) -> &mut TextureGenerator {
        &mut self.generator
    }

    /// Access the pipeline cache directly.
    pub fn pipeline_cache(&mut self) -> &mut PipelineCache {
        &mut self.pipeline_cache
    }

    /// Access the fullscreen renderer directly.
    pub fn fullscreen(&mut self) -> &mut FullscreenRenderer {
        &mut self.fullscreen
    }

    /// Render using procedural textures.
    ///
    /// Only sets up the pipeline and bind groups. Must call `pass.draw()` afterward yourself.
    ///
    /// Textures are automatically loaded based on `texture_keys`. Each key's `shader_id`
    /// is lowercased and `.wgsl` is appended to find the compute shader.
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
    /// Use this when you have texture views from sources other than the generator.
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
            .get(&(texture_views.len(), has_shadow))
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
            .get_or_create(shader_path, &bind_group_layout_refs, options);
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


    /// Render using custom bind group layouts.
    ///
    /// For advanced use cases where you need full control over bind groups.
    pub fn render_with_layouts(
        &mut self,
        shader_path: &Path,
        bind_group_layouts: &[&BindGroupLayout],
        bind_groups: &[&BindGroup],
        options: &PipelineOptions,
        pass: &mut RenderPass,
    ) {
        let pipeline = self.pipeline_cache.get_or_create(shader_path, bind_group_layouts, options);
        pass.set_pipeline(pipeline);

        for (i, bg) in bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *bg, &[]);
        }
    }

    /// Render a fullscreen debug visualization.
    pub fn render_fullscreen_debug(
        &mut self,
        texture: &TextureView,
        visualization: DebugVisualization,
        target_format: TextureFormat,
        msaa_samples: u32,
        pass: &mut RenderPass,
    ) {
        self.fullscreen.render(texture, visualization, target_format, msaa_samples, pass);
    }

    /// Update depth visualization parameters.
    pub fn update_depth_params(&mut self, params: DepthDebugParams) {
        self.fullscreen.update_depth_params(params);
    }

    /// Clear cached bind groups (call on window resize or texture recreation).
    pub fn invalidate_bind_groups(&mut self) {
        self.materials.clear();
        self.fullscreen.invalidate_bind_groups();
        self.uniform_bind_groups.clear();
    }

    /// Reload render shaders from disk.
    pub fn reload_render_shaders(&mut self, paths: &[PathBuf]) {
        self.pipeline_cache.reload_shaders(paths);
    }

    /// Reload texture generation shaders and clear texture cache.
    pub fn reload_texture_shaders(&mut self) {
        self.generator.reload_shaders();
    }

    /// Clear all caches (pipelines, textures, bind groups).
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
