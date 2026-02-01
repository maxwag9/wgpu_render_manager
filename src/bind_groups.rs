use std::collections::HashMap;
use wgpu::{AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Device, FilterMode, MipmapFilterMode, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, TextureSampleType, TextureView, TextureViewDimension};

#[derive(Clone, Hash, PartialEq, Eq)]
struct MaterialBindGroupKey(Vec<usize>);

impl MaterialBindGroupKey {
    fn from_views(views: &[&TextureView]) -> Self {
        Self(views.iter().map(|v| *v as *const TextureView as usize).collect())
    }
}

/// Manages material bind groups containing textures and samplers.
pub struct MaterialBindGroups {
    device: Device,
    sampler: Sampler,
    pub layouts: HashMap<usize, BindGroupLayout>,
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
    pub fn layout(&mut self, texture_count: usize) -> &BindGroupLayout {
        if !self.layouts.contains_key(&texture_count) {
            let entries: Vec<BindGroupLayoutEntry> = (0..texture_count)
                .flat_map(|i| {
                    let base = i as u32 * 2;
                    vec![
                        BindGroupLayoutEntry {
                            binding: base,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: base + 1,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Sampler(SamplerBindingType::Filtering),
                            count: None,
                        },
                    ]
                })
                .collect();

            let layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("material layout ({} textures)", texture_count)),
                entries: &entries,
            });

            self.layouts.insert(texture_count, layout);
        }

        self.layouts.get(&texture_count).unwrap()
    }

    /// Returns a bind group for the given texture views, creating it if necessary.
    pub fn get_or_create(&mut self, texture_views: &[&TextureView]) -> &BindGroup {
        let key = MaterialBindGroupKey::from_views(texture_views);
        let texture_count = texture_views.len();

        let _ = self.layout(texture_count);

        if !self.bind_groups.contains_key(&key) {
            let layout = self.layouts.get(&texture_count).unwrap();

            let entries: Vec<BindGroupEntry> = texture_views
                .iter()
                .enumerate()
                .flat_map(|(i, view)| {
                    let base = i as u32 * 2;
                    vec![
                        BindGroupEntry {
                            binding: base,
                            resource: BindingResource::TextureView(view),
                        },
                        BindGroupEntry {
                            binding: base + 1,
                            resource: BindingResource::Sampler(&self.sampler),
                        },
                    ]
                })
                .collect();

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