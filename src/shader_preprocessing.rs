use std::collections::HashSet;
use std::path::Path;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

/// Compiles a WGSL shader with a lightweight preprocessing step.
///
/// This function extends plain WGSL with a **compile-time preprocessing layer**
/// similar to `#ifdef` / `#include` in GLSL or C, while still producing **valid WGSL**
/// before it reaches wgpu/Naga.
///
/// ### Supported directives
///
/// The preprocessor understands the following directives:
///
/// - `#ifdef NAME`
/// - `#ifndef NAME`
/// - `#else`
/// - `#endif`
/// - `#include "relative_path.wgsl"`
///
/// These directives are evaluated **on the CPU** before shader compilation.
/// WGSL itself never sees them.
///
/// ### `defines`
///
/// The `defines` set controls which `#ifdef` / `#ifndef` blocks are active.
/// Each entry represents a boolean compile-time flag.
///
/// Example:
/// ```text
/// #ifdef MSAA
///     var depth: texture_depth_multisampled_2d;
/// #else
///     var depth: texture_depth_2d;
/// #endif
/// ```
///
/// Passing `"MSAA"` in `defines` will select the multisampled path.
/// Otherwise, the non-MSAA path is emitted.
///
/// This allows:
/// - Switching **binding types** (which WGSL cannot do dynamically)
/// - Zero runtime branching
/// - One shader source producing multiple optimized pipelines
///
/// ### `#include` behavior
///
/// `#include` recursively inlines other `.wgsl` files using paths
/// relative to the including file. Included files may themselves
/// contain preprocessing directives.
///
/// This is typically used for:
/// - Shared structs and constants
/// - Common math utilities
/// - Render parameter blocks
///
/// ### Important notes
///
/// - This is **not** runtime logic. All decisions are made at compile time.
/// - Different `defines` produce **different shader modules**.
/// - Binding layouts must still match the selected shader variant.
/// - The preprocessing step is intentionally simple and line-based.
///
/// ### Typical use cases
///
/// - MSAA vs non-MSAA depth sampling
/// - Feature toggles (shadows, AO, fog variants)
/// - Platform-specific bindings
/// - Pass fusion without shader duplication
///
/// ### Panics
///
/// Panics if the shader file or any included file cannot be read.
pub(crate) fn compile_wgsl(
    device: &Device,
    path: &Path,
    defines: &HashSet<String>,
) -> ShaderModule {
    let source = std::fs::read_to_string(path).unwrap();
    let processed = preprocess_wgsl(path, &source, defines);
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(path.to_str().unwrap()),
        source: ShaderSource::Wgsl(processed.into()),
    })
}

fn preprocess_wgsl(
    base: &Path,
    src: &str,
    defines: &HashSet<String>,
) -> String {
    let mut stack: Vec<bool> = vec![];

    src.lines()
        .flat_map(|line| {
            let t = line.trim();

            if let Some(k) = t.strip_prefix("#ifdef ") {
                stack.push(defines.contains(k.trim()));
                return None;
            }
            if let Some(k) = t.strip_prefix("#ifndef ") {
                stack.push(!defines.contains(k.trim()));
                return None;
            }
            if t.starts_with("#else") {
                let v = stack.last_mut().unwrap();
                *v = !*v;
                return None;
            }
            if t.starts_with("#endif") {
                stack.pop();
                return None;
            }

            if stack.iter().any(|v| !*v) {
                return None;
            }

            if let Some(p) = t.strip_prefix("#include \"") {
                let p = p.strip_suffix('"').unwrap();
                let mut inc = base.parent().unwrap().to_path_buf();
                inc.push(p);
                let s = std::fs::read_to_string(&inc).unwrap();
                return Some(preprocess_wgsl(&inc, &s, defines));
            }

            Some(line.to_string())
        })
        .collect::<Vec<_>>()
        .join("\n")
}
