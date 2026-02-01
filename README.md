# wgpu-crm-mgr

An easy way to use wgpu without layout, pipeline and bindgroup hell. Cached and lightweight.
Cache-driven render and compute manager for `wgpu`.
Made primarily for my own Game, [Rusty Skylines](https://github.com/maxwag9/rusty_skylines).

Greatly increases your productivity, because it removes immense amounts of the boilerplate from manual:
- pipeline creation
- bind group layout management
- bind group caching
- MSAA-aware texture binding
- fullscreen passes
- procedural textures via compute

The goal:  
You describe what you want to bind, and the manager figures out the rest.

ALSO includes an automatic procedural texture manager used by the RenderManager, 
which makes textures using compute shaders automatically and binds them automatically!

# Examples
### Compute

```rust
    let mut compute = ComputeSystem::new(&self.device, &self.queue); // Caches device and queue so you don't have to pass it in again
    compute.compute(
        "Resolve Depth",             // Label
        vec![&msaa_depth_view],      // Input Views
        vec![&resoloved_depth_view], // Output Views
        shader_path,                 // Shader Path
        options,                     // ComputePipelineOptions
        &[&camera_buffer]            // Optional Buffers
    );
```

### Render

```rust
    let mut render_manager = RenderManager::new(&device, &queue, texture_shaders_path);
    render_manager.render(
        &[TextureKey],               // TextureKeys for included procedural textures
        shader_path,                 // Shader Path
        options,                     // PipelineOptions
        &[&camera_buffer],           // Optional Buffers
        pass                         // Your own pass!
    );
```

## What this crate is

- A **low-level helper**, not a framework
- Designed for **engines and tools**, not examples
- Focused on **performance, correctness, and zero magic state**
- Explicit data flow, no global settings, no hidden stuff

## Features

- Automatic render and compute pipeline caching
- Bind group layouts inferred from usage
- MSAA-safe texture handling
- Unified render + compute architecture
- Fullscreen render helpers
- Procedural texture generation using compute shaders
- No engine-specific globals or renderer state


## Non-goals

This crate intentionally does **not**:
- manage windowing or surfaces
- own frame graphs
- impose a render architecture
- hide wgpu concepts

You still control your render passes, encoders, and frame timing!


## Philosophy

wgpu is explicit.  
So of course this crate stays explicit! But it removes repetition.

If something can be:
- derived
- cached
- reused

…it probably is…


## Status

This is an early release (`0.1.x`), so APIs will evolve as I develop my Game, [Rusty Skylines](https://github.com/maxwag9/rusty_skylines).

- Feedback welcome


MIT License, because this crate WILL HELP YOU!
