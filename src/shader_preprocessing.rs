use std::collections::{HashMap};
use std::path::Path;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

/// Compiles a WGSL shader with a lightweight preprocessing step.
///
/// This function extends plain WGSL with a **compile-time preprocessing layer**
/// similar to `#ifdef` / `#include` in GLSL or C, while still producing **valid WGSL**
/// before it reaches wgpu/Naga.
///
/// # Updating Defines
/// You MUST update the defines using [`update_define()`](crate::renderer::RenderManager::update_define()) in the [`RenderManager`](crate::renderer::RenderManager)!
///
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
/// In this example, passing `"MSAA"` in `defines` will select the multisampled path.
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
/// Also panics if any point of the preprocessing fails or the define doesn't exist if defined in the shader.
pub fn compile_wgsl(
    device: &Device,
    path: &Path,
    defines: &HashMap<String, bool>,
) -> ShaderModule {
    let source = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("Failed to read shader file {}: {}", path.display(), e)
    });

    let mut conditional_stack: Vec<bool> = vec![];
    let mut processed = preprocess_wgsl(path, &source, defines, &mut conditional_stack);

    if !conditional_stack.is_empty() {
        panic!(
            "Unbalanced preprocessing directives in shader {} (or one of its #included files): open #ifdef/#ifndef without matching #endif",
            path.display()
        );
    }

    if contains_wgsl_identifier(&processed, "binding_array")
        && !has_wgsl_enable_directive(&processed, "wgpu_binding_array")
    {
        processed.insert_str(0, "enable wgpu_binding_array;\n");
    }

    let label_str = path
        .to_str()
        .unwrap_or_else(|| panic!("Shader path {} is not valid UTF-8", path.display()));

    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label_str),
        source: ShaderSource::Wgsl(processed.into()),
    })
}

fn preprocess_wgsl(
    path: &Path,
    src: &str,
    defines: &HashMap<String, bool>,
    stack: &mut Vec<bool>,
) -> String {
    let processed_lines: Vec<String> = src
        .lines()
        .enumerate()
        .flat_map(|(i, line)| {
            let line_num = i + 1;
            let t = line.trim();

            // --- #ifdef ---
            if let Some(rest) = t.strip_prefix("#ifdef ") {
                let name = rest.trim();
                let value = *defines.get(name).expect(&format!(
                    "{}:{}: Unknown preprocessing define '{}' in #ifdef",
                    path.display(),
                    line_num,
                    name
                ));
                stack.push(value);
                return vec![];
            }

            // --- #ifndef ---
            if let Some(rest) = t.strip_prefix("#ifndef ") {
                let name = rest.trim();
                let value = *defines.get(name).expect(&format!(
                    "{}:{}: Unknown preprocessing define '{}' in #ifndef",
                    path.display(),
                    line_num,
                    name
                ));
                stack.push(!value);
                return vec![];
            }

            // --- #else ---
            if t.starts_with("#else") {
                let v = stack.last_mut().expect(&format!(
                    "{}:{}: #else without matching #ifdef/#ifndef",
                    path.display(),
                    line_num
                ));
                *v = !*v;
                return vec![];
            }

            // --- #endif ---
            if t.starts_with("#endif") {
                if stack.pop().is_none() {
                    panic!(
                        "{}:{}: #endif without matching #ifdef/#ifndef",
                        path.display(),
                        line_num
                    );
                }
                return vec![];
            }

            // --- Skip lines in inactive blocks ---
            if stack.iter().any(|&v| !v) {
                return vec![];
            }

            // --- #include ---
            if let Some(rest) = t.strip_prefix("#include \"") {
                let p = rest.strip_suffix('"').expect(&format!(
                    "{}:{}: Malformed #include directive: missing closing quote",
                    path.display(),
                    line_num
                ));

                let parent = path.parent().expect("Cannot resolve relative #include: shader path has no parent directory");
                let mut inc_path = parent.to_path_buf();
                inc_path.push(p);

                let inc_source = std::fs::read_to_string(&inc_path).unwrap_or_else(|e| {
                    panic!(
                        "{}:{}: Failed to read included shader \"{}\" (resolved to {}): {}",
                        path.display(),
                        line_num,
                        p,
                        inc_path.display(),
                        e
                    );
                });

                let inc_processed = preprocess_wgsl(&inc_path, &inc_source, defines, stack);

                return inc_processed
                    .lines()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
            }

            // --- Normal line ---
            vec![line.to_string()]
        })
        .collect();

    processed_lines.join("\n") + "\n"
}
fn contains_wgsl_identifier(source: &str, identifier: &str) -> bool {
    wgsl_tokens(source)
        .into_iter()
        .any(|token| token == identifier.as_bytes())
}

fn has_wgsl_enable_directive(source: &str, extension: &str) -> bool {
    wgsl_tokens(source).windows(3).any(|tokens| {
        tokens[0] == b"enable"
            && tokens[1] == extension.as_bytes()
            && tokens[2] == b";"
    })
}

fn wgsl_tokens(source: &str) -> Vec<&[u8]> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        if bytes[i] == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                i += 2;

                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }

                continue;
            }

            if bytes[i + 1] == b'*' {
                i += 2;
                let mut depth = 1usize;

                while i < bytes.len() && depth > 0 {
                    if i + 1 < bytes.len() {
                        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                            depth += 1;
                            i += 2;
                            continue;
                        }

                        if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                            depth -= 1;
                            i += 2;
                            continue;
                        }
                    }

                    i += 1;
                }

                continue;
            }
        }

        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            i += 1;

            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
            }

            tokens.push(&bytes[start..i]);
            continue;
        }

        tokens.push(&bytes[i..i + 1]);
        i += 1;
    }

    tokens
}