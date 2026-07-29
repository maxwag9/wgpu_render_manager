struct DownsampleParams {
    alpha_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: DownsampleParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(dst);
    if (gid.x >= dst_size.x || gid.y >= dst_size.y) {
        return;
    }

    let base = vec2<i32>(gid.xy) * 2;
    let c00 = textureLoad(src, base + vec2<i32>(0, 0), 0);
    let c10 = textureLoad(src, base + vec2<i32>(1, 0), 0);
    let c01 = textureLoad(src, base + vec2<i32>(0, 1), 0);
    let c11 = textureLoad(src, base + vec2<i32>(1, 1), 0);

    let sum_alpha = c00.a + c10.a + c01.a + c11.a;

    var avg_color: vec3<f32>;
    if (sum_alpha > 0.0001) {
        avg_color = (c00.rgb * c00.a + c10.rgb * c10.a + c01.rgb * c01.a + c11.rgb * c11.a) / sum_alpha;
    } else {
        avg_color = (c00.rgb + c10.rgb + c01.rgb + c11.rgb) * 0.25;
    }

    let avg_alpha = sum_alpha * 0.25;
    let out_alpha = clamp(avg_alpha * params.alpha_scale, 0.0, 1.0);

    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(avg_color, out_alpha));
}