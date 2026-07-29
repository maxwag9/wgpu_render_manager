struct CoverageParams {
    threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<uniform> params: CoverageParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(src);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let texel = textureLoad(src, vec2<i32>(gid.xy), 0);
    if (texel.a > params.threshold) {
        atomicAdd(&counter, 1u);
    }
}