use std::collections::HashMap;
use wgpu::{AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Device, FilterMode, MipmapFilterMode, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, TextureSampleType, TextureView, TextureViewDimension};

#[derive(Clone, Hash, PartialEq, Eq)]
struct MaterialBindGroupKey {
    view_ids: Vec<usize>,
    has_shadow: bool
}

impl MaterialBindGroupKey {
    fn from_views(views: &[&TextureView], has_shadow: bool) -> Self {
        Self { view_ids: views.iter().map(|v| *v as *const TextureView as usize).collect(), has_shadow }
    }
}

/// Manages material bind groups containing textures and samplers.
pub struct MaterialBindGroups {
    device: Device,
    sampler: Sampler,
    pub layouts: HashMap<(usize, bool), BindGroupLayout>,
    bind_groups: HashMap<MaterialBindGroupKey, BindGroup>,
}

impl MaterialBindGroups {
    pub fn new(device: Device) -> Self {
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("material sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            device,
            sampler,
            layouts: HashMap::new(),
            bind_groups: HashMap::new(),
        }
    }

    /// Returns the bind group layout for the given texture count.
    pub fn layout(
        &mut self,
        texture_views: &[&TextureView],
        has_shadow: bool,
    ) -> &BindGroupLayout {
        let key = (texture_views.len(), has_shadow);

        if !self.layouts.contains_key(&key) {
            let mut entries = Vec::new();
            let mut binding = 0;

            // 0: material sampler
            entries.push(BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            });
            binding += 1;

            // 1..N: textures (auto-detect)
            for view in texture_views {
                let tex = view.texture();
                let is_depth = tex.format().has_depth_aspect();

                entries.push(BindGroupLayoutEntry {
                    binding,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: tex.sample_count() > 1,
                        view_dimension: if tex.depth_or_array_layers() > 1 {
                            TextureViewDimension::D2Array
                        } else {
                            TextureViewDimension::D2
                        },
                        sample_type: if is_depth {
                            TextureSampleType::Depth
                        } else {
                            TextureSampleType::Float { filterable: true }
                        },
                    },
                    count: None,
                });

                binding += 1;
            }

            // Shadow (optional)
            if has_shadow {
                entries.push(BindGroupLayoutEntry {
                    binding,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                });
                binding += 1;

                entries.push(BindGroupLayoutEntry {
                    binding,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2Array,
                        sample_type: TextureSampleType::Depth,
                    },
                    count: None,
                });
            }

            let layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("material bind group layout"),
                entries: &entries,
            });

            self.layouts.insert(key, layout);
        }

        self.layouts.get(&key).unwrap()
    }


    /// Returns a bind group for the given texture views, creating it if necessary.
    pub fn get_or_create(
        &mut self,
        texture_views: &[&TextureView],
        shadow: Option<(&Sampler, &TextureView)>,
    ) -> &BindGroup {
        let has_shadow = shadow.is_some();

        let key = MaterialBindGroupKey::from_views(texture_views, has_shadow);

        if !self.bind_groups.contains_key(&key) {
            // Ensure layout exists
            let layout = &self.layout(texture_views, has_shadow).clone();

            let mut entries: Vec<BindGroupEntry> = Vec::new();
            let mut binding: u32 = 0;

            // binding 0: material sampler
            entries.push(BindGroupEntry {
                binding,
                resource: BindingResource::Sampler(&self.sampler),
            });
            binding += 1;

            // binding 1..N: textures
            for view in texture_views {
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::TextureView(view),
                });
                binding += 1;
            }

            // optional shadow
            if let Some((shadow_sampler, shadow_view)) = shadow {
                // comparison sampler
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::Sampler(shadow_sampler),
                });
                binding += 1;

                // depth texture array
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::TextureView(shadow_view),
                });
            }

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("material bind group"),
                layout,
                entries: &entries,
            });

            self.bind_groups.insert(key.clone(), bind_group);
        }

        self.bind_groups.get(&key).unwrap()
    }

    /// Clears all cached bind groups.
    pub fn clear(&mut self) {
        self.bind_groups.clear();
    }
}