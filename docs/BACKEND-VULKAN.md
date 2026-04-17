# Shannon-Prime: Vulkan Backend

## Overview

The Vulkan backend runs VHT2 compression via compute shaders, targeting cross-platform GPU acceleration. It works on NVIDIA, AMD, Intel, and Qualcomm Adreno GPUs — any device with Vulkan 1.1+ compute support. When no Vulkan device is available, it falls back to the C core implementation transparently.

**Current runtime note:** Vulkan compute is currently routed through the CPU staged VHT2 inside the host wrappers (`vk_compress_one` / `vk_decompress_one`) because the GPU dispatch hits `VK_ERROR_DEVICE_LOST` at the first `vkWaitForFences` on RTX 2060. Pipelines are still created so `test_vulkan` exercises the full init path; only the GPU submit is skipped. Debugging the shader hang is tracked separately and needs RenderDoc on the target GPU. The wording below describes the shader pipeline as it is wired up and as it will execute once the hang is resolved.

## Compute Shaders

The following GLSL compute shaders live in `backends/vulkan/shaders/`:

**`vilenkin.comp`:** In-place VHT2 butterfly (self-inverse, 1/√p per stage, no 1/N on reconstruction). One workgroup per vector, shared memory for butterfly passes. The workgroup size is a specialization constant matching head_dim (64, 128, or 256). For hd=128 at p=2: 7 barrier-synchronized passes in shared memory, then writeback. The legacy `wht.comp` shader has been removed; `vilenkin.comp` is the single transform.

**`mobius_reorder.comp`:** Gather-based Möbius permutation. Each thread reads from `order[tid]` and writes to position `tid`. The permutation table is a storage buffer bound at descriptor set 0, binding 2.

**`band_quantize.comp`:** Per-band abs-max reduction in shared memory, then sequential bit packing by thread 0. Push constants carry the band configuration (n_bands, bits per band, band_size, total output words).

**`band_dequantize.comp`:** Inverse of `band_quantize.comp`. Reads the packed payload + per-band fp16 scale, unpacks to signed integers, and scales back to fp32. Mirrors the CPU core's non-finite-scale guard: if an fp16 scale round-trips to Inf/NaN the whole band decodes as zero instead of poisoning the inverse VHT2.

**`knight_predict.comp`:** Sqfree+spinor Knight CSR predict shader used by the aggressive path (also currently CPU-staged until the `VK_ERROR_DEVICE_LOST` issue is resolved).

## Pipeline

```
Vulkan Compute Pipeline
════════════════════════

  Input Buffer (raw K/V, fp32)
       │
       ▼
  ┌─────────────┐    Dispatch 1: vilenkin.comp
  │   VHT2      │    workgroup = (head_dim,), groups = n_vectors
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    Dispatch 2: mobius_reorder.comp
  │   Möbius     │    (K only — V skips this step)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    Dispatch 3: band_quantize.comp
  │   Quantize   │    Output to compressed cache buffer
  └──────┬──────┘
         │
         ▼
  Compressed Cache Buffer
```

Read path reverses: dequantize → unreorder → VHT2 (reusing the same `vilenkin.comp` shader since VHT2 is self-inverse; no separate inverse shader, no 1/N on reconstruction).

## Building

Compile shaders to SPIR-V:

```bash
glslangValidator -V backends/vulkan/shaders/vilenkin.comp -o backends/vulkan/shaders/vilenkin.spv
glslangValidator -V backends/vulkan/shaders/mobius_reorder.comp -o backends/vulkan/shaders/mobius_reorder.spv
glslangValidator -V backends/vulkan/shaders/band_quantize.comp -o backends/vulkan/shaders/band_quantize.spv
glslangValidator -V backends/vulkan/shaders/band_dequantize.comp -o backends/vulkan/shaders/band_dequantize.spv
glslangValidator -V backends/vulkan/shaders/knight_predict.comp -o backends/vulkan/shaders/knight_predict.spv
```

Compile the host code (requires Vulkan SDK for full GPU path):

```bash
gcc -O2 -DVK_USE_PLATFORM -o build/test_vulkan \
    tests/test_vulkan.c \
    backends/vulkan/shannon_prime_vulkan.c \
    core/shannon_prime.c \
    -lvulkan -lm
```

Without Vulkan SDK, the host code compiles with CPU fallback (all 4 tests pass):

```bash
gcc -O2 -o build/test_vulkan \
    tests/test_vulkan.c \
    backends/vulkan/shannon_prime_vulkan.c \
    core/shannon_prime.c \
    -lm
```

## Integration Modes

**Standalone:** Creates its own VkDevice and VkQueue. For testing and benchmarking.

```c
sp_vulkan_cache_t *cc;
sp_vulkan_cache_init(&cc, &cfg, max_seq_len, NULL, NULL);
```

**Shared:** Uses the existing Vulkan device from llama.cpp's Vulkan backend or another inference engine. Zero overhead from device creation.

```c
sp_vulkan_cache_init(&cc, &cfg, max_seq_len, existing_vk_device, existing_vk_queue);
```

## Buffer API

For zero-copy integration when K/V are already in Vulkan buffers:

```c
// Write from existing VkBuffer (no CPU→GPU transfer)
sp_vulkan_write_k_buffer(cc, layer, head, pos, vk_buffer, byte_offset);

// Read into existing VkBuffer (stays on GPU)
sp_vulkan_read_k_buffer(cc, layer, head, pos, vk_buffer, byte_offset);
```

## Adreno-Specific Notes

On Qualcomm Adreno GPUs (Snapdragon 8 Gen 1 / Adreno 730):

- Shared memory per workgroup: 32 KB (sufficient for hd=256)
- Max workgroup size: 1024 threads (sufficient for any head_dim)
- Wave64 execution: 64 threads per warp (vs 32 on NVIDIA)
- OpenCL is also available but Vulkan compute is preferred for llama.cpp integration

The Vulkan backend is the GPU path for mobile inference. The Adreno backend handles the CPU/NEON writeback path for when the GPU is busy with inference.
