// lib.rs
//! # wgpu_render_manager
//!
//! A helper crate for rendering and computing using `wgpu`, made specifically for game engines.
//!
//! It can:
//! - Render textures to the screen for debugging and visualization purposes
//! - Render anything using automatically procedurally generated textures
//! - Procedurally generate textures using TextureKey and a shader
//! - Make compute pipelines trivial using [`compute()`](compute_system::ComputeSystem::compute())
//!
//! This crate makes game development and rendering with fullscreen passes a breeze.
//!
//! ## Goals
//! - Minimal boilerplate
//! - Zero magic around bind groups
//! - Easy fullscreen rendering of any texture
//!
//! ## Shader Binding layout
//! - `@group(0) @binding(0)`: trilinear sampler
//! - `@group(0) @binding(0..n)`: textures as texture_2d<f32> or texture_multisampled_2d<f32>
//! - `@group(0) @binding(n+1)`: (optional) shadow_sampler
//! - `@group(0) @binding(n+2)`: (optional) shadow textures as texture_depth_2d_array
//! - `@group(1) @binding(0..n)`: uniforms, in the same order as input
//!
//! ## Basic usage
//! ```no_run
//! // Inside a render pass
//! render_manager.render_with_textures(
//!     &texture_views.as_slice(), // Texture Views
//!     shader_path.as_path(),     // Shader Path
//!     options,                   // PipelineOptions
//!     &[&uniforms_buffer],       // Buffers
//!     &mut render_pass,          // Render Pass
//! );
//! ```
//!
//! Used in my game [Rusty Skylines](https://github.com/maxwag9/rusty_skylines)

pub mod compute_system;
pub mod generator;
pub mod pipelines;
pub mod fullscreen;
pub mod renderer;
mod bind_groups;
mod shader_preprocessing;
