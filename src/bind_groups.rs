use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Device, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, TextureAspect, TextureSampleType, TextureView, TextureViewDimension};

#[derive(Clone, Hash, PartialEq, Eq)]
struct MaterialBindGroupKey {
    views_hash: u64,
    has_shadow: bool,
    texture_array_hash: u64,
}

impl MaterialBindGroupKey {
    fn from_views(views: &[&TextureView], has_shadow: bool, texture_array_view: &TextureView) -> Self {
        let mut hasher = DefaultHasher::new();
        for v in views {
            v.hash(&mut hasher);
        }
        let texture_array_hash = {
            let mut h = DefaultHasher::new();
            texture_array_view.hash(&mut h);
            h.finish()
        };
        Self { views_hash: hasher.finish(), has_shadow, texture_array_hash }
    }
}
#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct LayoutKey {
    layout_hash: u64,
    has_shadow: bool
}

impl LayoutKey {
    pub(crate) fn from_views(views: &[&TextureView], has_shadow: bool) -> Self {
        let mut hasher = DefaultHasher::new();
        for v in views {
            v.hash(&mut hasher);
        }
        Self {
            layout_hash: hasher.finish(),
            has_shadow
        }
    }
}

/// Manages material bind groups containing textures and samplers.
pub(crate) struct MaterialBindGroups {
    device: Device,
    pub(crate) layouts: HashMap<LayoutKey, BindGroupLayout>,
    bind_groups: HashMap<MaterialBindGroupKey, BindGroup>,
    samplers: HashMap<u64, Sampler>
}

impl MaterialBindGroups {
    pub(crate) fn new(device: Device) -> Self {
        let samplers = HashMap::new();

        Self {
            device,
            layouts: HashMap::new(),
            bind_groups: HashMap::new(),
            samplers
        }
    }

    /// Returns the bind group layout for the given texture count.
    pub(crate) fn layout(
        &mut self,
        texture_views: &[&TextureView],
        has_shadow: bool
    ) -> &BindGroupLayout {
        let key = LayoutKey::from_views(texture_views, has_shadow);

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

            // 1: texture array
            entries.push(BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: TextureViewDimension::D2Array,
                    sample_type: TextureSampleType::Float { filterable: true },
                },
                count: None,
            });
            binding += 1;


            let device_features = self.device.features();
            // 2..N: individual textures
            for view in texture_views {
                let tex = view.texture();
                let format = tex.format();
                let is_multisampled = tex.sample_count() > 1;

                let sample_type = format
                    .sample_type(Some(TextureAspect::All), Some(device_features))
                    .or_else(|| format.sample_type(Some(TextureAspect::DepthOnly), Some(device_features)))
                    .expect("Unsupported texture format");

                let sample_type = if is_multisampled {
                    match sample_type {
                        TextureSampleType::Float { .. } => TextureSampleType::Float { filterable: false },
                        other => other,
                    }
                } else {
                    sample_type
                };

                entries.push(BindGroupLayoutEntry {
                    binding,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: is_multisampled,
                        view_dimension: if tex.depth_or_array_layers() > 1 {
                            TextureViewDimension::D2Array
                        } else {
                            TextureViewDimension::D2
                        },
                        sample_type,
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

            self.layouts.insert(key.clone(), layout);
        }

        self.layouts.get(&key).unwrap()
    }

    /// Returns a bind group for the given texture views, creating it if necessary.
    pub(crate) fn get_or_create(
        &mut self,
        texture_views: &[&TextureView],
        shadow: Option<(&Sampler, &TextureView)>,
        sampler: &SamplerDescriptor,
        texture_array_view: &TextureView,
    ) -> &BindGroup {
        let has_shadow = shadow.is_some();

        let key = MaterialBindGroupKey::from_views(texture_views, has_shadow, texture_array_view);

        if !self.bind_groups.contains_key(&key) {
            let layout = &self.layout(texture_views, has_shadow).clone();

            let mut entries: Vec<BindGroupEntry> = Vec::new();
            let mut binding: u32 = 0;

            // binding 0: material sampler
            let sampler = self.get_or_create_sampler(sampler).clone();
            entries.push(BindGroupEntry {
                binding,
                resource: BindingResource::Sampler(&sampler),
            });
            binding += 1;

            // binding 1: texture array (if present)
            entries.push(BindGroupEntry {
                binding,
                resource: BindingResource::TextureView(texture_array_view),
            });
            binding += 1;


            // binding 2..N: individual textures
            for view in texture_views {
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::TextureView(view),
                });
                binding += 1;
            }

            // optional shadow
            if let Some((shadow_sampler, shadow_view)) = shadow {
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::Sampler(shadow_sampler),
                });
                binding += 1;

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

    fn get_or_create_sampler(&mut self, sampler_descriptor: &SamplerDescriptor) -> &Sampler {
        let key = hash_sampler(sampler_descriptor);

        if !self.samplers.contains_key(&key) {
            let sampler = self.device.create_sampler(sampler_descriptor);
            
            self.samplers.insert(key.clone(), sampler);
        }
        self.samplers.get(&key).unwrap()
    }
}

fn hash_sampler(sampler_descriptor: &SamplerDescriptor) -> u64 {
    let sd = sampler_descriptor;
    let hasher = &mut DefaultHasher::new();
    sd.label.hash(hasher);
    sd.address_mode_u.hash(hasher);
    sd.address_mode_v.hash(hasher);
    sd.address_mode_w.hash(hasher);
    sd.anisotropy_clamp.hash(hasher);
    sd.border_color.hash(hasher);
    sd.compare.hash(hasher);
    sd.lod_max_clamp.to_bits().hash(hasher);
    sd.lod_min_clamp.to_bits().hash(hasher);
    sd.mag_filter.hash(hasher);
    sd.min_filter.hash(hasher);
    sd.mipmap_filter.hash(hasher);

    hasher.finish()
}