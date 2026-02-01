// lib.rs
//! GPU-accelerated procedural texture generation using wgpu compute shaders.
//!
//! This crate provides utilities for generating procedural textures on the GPU,
//! with automatic caching and mipmap generation.
//!
//! ## Example
//!
//! ```ignore
//! use procedural_textures::{TextureGenerator, TextureParams, ShaderSource};
//!
//! let mut generator = TextureGenerator::new(device, queue);
//!
//! // Register a compute shader
//! generator.register("noise", ShaderSource::Wgsl(include_str!("noise.wgsl")));
//!
//! // Generate a texture
//! let key = TextureKey::new("noise", TextureParams::default().with_seed(42), 512);
//! let view = generator.get_or_create(&key);
//! ```
//! Also features an automatic, extremely powerful compute pass system for textures.

pub mod compute_system;
pub mod generator;
pub mod pipelines;
pub mod fullscreen;
pub mod renderer;
pub mod bind_groups;
