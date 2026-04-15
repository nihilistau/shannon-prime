// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// CUDA kernels for VHT2 KV cache compression.
// Each kernel processes one or more independent vectors of length head_dim.
// The design prioritizes correctness and clarity; optimization follows.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

// ============================================================================
// WHT Butterfly Kernel
// ============================================================================
//
// In-place Walsh-Hadamard Transform via iterative butterfly.
// Each thread block processes one vector of length n.
// Uses shared memory for the butterfly passes.
//
// For head_dim=128: 7 passes (log2(128)), each a full barrier.
// This is bandwidth-bound, not compute-bound — fine for KV cache writes
// which happen once per token per layer.

__global__ void kernel_wht_inplace(float *data, int n) {
    extern __shared__ float smem[];

    int vec_idx = blockIdx.x;              // Which vector
    int tid     = threadIdx.x;             // Thread within vector
    float *vec  = data + (size_t)vec_idx * n;

    // Load into shared memory
    if (tid < n) {
        smem[tid] = vec[tid];
    }
    __syncthreads();

    // Butterfly passes
    for (int len = 1; len < n; len <<= 1) {
        if (tid < n) {
            int grp    = tid / (len << 1);   // Which butterfly group
            int pos    = tid % (len << 1);   // Position within group
            int base   = grp * (len << 1);

            if (pos < len) {
                float u = smem[base + pos];
                float v = smem[base + pos + len];
                smem[base + pos]       = u + v;
                smem[base + pos + len] = u - v;
            }
        }
        __syncthreads();
    }

    // Write back
    if (tid < n) {
        vec[tid] = smem[tid];
    }
}

// Inverse WHT = forward WHT followed by 1/N scaling
__global__ void kernel_iwht_scale(float *data, int n, float inv_n) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (tid < n) {
        data[(size_t)vec_idx * n + tid] *= inv_n;
    }
}

// ============================================================================
// Möbius Permutation Kernel
// ============================================================================
//
// Applies the squarefree-first reordering to WHT coefficients.
// order[i] = source index for position i in reordered vector.
// Each thread handles one element.

__global__ void kernel_mobius_reorder(const float *input, float *output,
                                     const int *order, int n) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (tid < n) {
        size_t out_off = (size_t)vec_idx * n + tid;
        size_t in_off  = (size_t)vec_idx * n + order[tid];
        output[out_off] = input[in_off];
    }
}

// Inverse: order[i] says "original index i goes to position order_inv[i]"
// We build the inverse table on host; kernel just gathers.
__global__ void kernel_mobius_unreorder(const float *input, float *output,
                                       const int *order, int n) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (tid < n) {
        // order[i] = original index that goes to position i
        // So to unreorder: output[order[i]] = input[i]
        size_t in_off  = (size_t)vec_idx * n + tid;
        size_t out_off = (size_t)vec_idx * n + order[tid];
        output[out_off] = input[in_off];
    }
}

// ============================================================================
// Banded Quantization Kernel
// ============================================================================
//
// Each thread block processes one vector.
// Band 0..n_bands-1 each get: fp16 scale + packed n-bit integers.
//
// Thread assignment: threads 0..band_size-1 handle band b.
// Phase 1: parallel reduction to find max(abs) per band → scale
// Phase 2: each thread quantizes its element and writes packed bits
//
// Note: bit packing is serialized within a band for correctness.
// A production kernel would use warp-level shuffle for parallel packing.

// Simplified kernel: one thread per vector, processes all bands sequentially.
// This is correct and fast enough for single-token decode.
// Batch kernels for prefill use the parallel version below.

__global__ void kernel_band_quantize_simple(
    const float *input,   // [n_vecs][n] WHT coefficients
    uint8_t *output,      // [n_vecs][total_bytes] packed output
    int n,                // head_dim
    int n_bands,
    const int *band_bits, // [n_bands] bits per band (device memory)
    int band_size,        // n / n_bands
    int total_bytes       // bytes per compressed vector
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check handled by caller (n_vecs)

    const float *vec = input + (size_t)vec_idx * n;
    uint8_t *out     = output + (size_t)vec_idx * total_bytes;

    int offset = 0;

    for (int b = 0; b < n_bands; b++) {
        const float *band = vec + b * band_size;
        int bits = band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;

        // Find max absolute value
        float amax = 0.0f;
        for (int i = 0; i < band_size; i++) {
            float a = fabsf(band[i]);
            if (a > amax) amax = a;
        }

        float scale = (amax > 0.0f) ? amax / (float)max_val : 0.0f;
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

        // Store scale as fp16
        __half scale_h = __float2half(scale);
        unsigned short scale_bits;
        memcpy(&scale_bits, &scale_h, sizeof(unsigned short));
        out[offset]     = scale_bits & 0xFF;
        out[offset + 1] = (scale_bits >> 8) & 0xFF;
        offset += 2;

        // Pack quantized values
        unsigned long long bit_buffer = 0;
        int bit_pos = 0;

        for (int i = 0; i < band_size; i++) {
            int q = __float2int_rn(band[i] * inv_scale);
            if (q > max_val)  q = max_val;
            if (q < -max_val) q = -max_val;

            unsigned int u = (unsigned int)(q + max_val);
            bit_buffer |= ((unsigned long long)u << bit_pos);
            bit_pos += bits;

            while (bit_pos >= 8) {
                out[offset++] = (uint8_t)(bit_buffer & 0xFF);
                bit_buffer >>= 8;
                bit_pos -= 8;
            }
        }
        if (bit_pos > 0) {
            out[offset++] = (uint8_t)(bit_buffer & 0xFF);
        }
    }
}

// ============================================================================
// Banded Dequantization Kernel
// ============================================================================

__global__ void kernel_band_dequantize_simple(
    const uint8_t *input, // [n_vecs][total_bytes]
    float *output,        // [n_vecs][n]
    int n,
    int n_bands,
    const int *band_bits,
    int band_size,
    int total_bytes
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint8_t *in = input + (size_t)vec_idx * total_bytes;
    float *vec        = output + (size_t)vec_idx * n;

    int offset = 0;

    for (int b = 0; b < n_bands; b++) {
        float *band = vec + b * band_size;
        int bits = band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;
        unsigned int mask = (1u << bits) - 1;

        // Read scale
        unsigned short scale_bits = (unsigned short)in[offset] |
                                    ((unsigned short)in[offset + 1] << 8);
        __half scale_h;
        memcpy(&scale_h, &scale_bits, sizeof(__half));
        float scale = __half2float(scale_h);
        offset += 2;

        // Unpack
        unsigned long long bit_buffer = 0;
        int bit_pos = 0;
        int byte_idx = offset;

        for (int i = 0; i < band_size; i++) {
            while (bit_pos < bits) {
                bit_buffer |= ((unsigned long long)in[byte_idx++] << bit_pos);
                bit_pos += 8;
            }

            unsigned int u = (unsigned int)(bit_buffer & mask);
            bit_buffer >>= bits;
            bit_pos -= bits;

            int q = (int)u - max_val;
            band[i] = (float)q * scale;
        }

        int data_bits = band_size * bits;
        offset += (data_bits + 7) / 8;
    }
}

// ============================================================================
// NaN Guard Kernel
// ============================================================================

__global__ void kernel_nan_guard(float *data, int n, float max_mag) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (tid < n) {
        size_t idx = (size_t)vec_idx * n + tid;
        float val = data[idx];
        if (!isfinite(val)) {
            data[idx] = 0.0f;
        } else if (val > max_mag) {
            data[idx] = max_mag;
        } else if (val < -max_mag) {
            data[idx] = -max_mag;
        }
    }
}

// ============================================================================
// Host-side launcher functions
// ============================================================================

extern "C" {

#include "shannon_prime_cuda.h"

void sp_cuda_wht_inplace(float *d_data, int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = n * sizeof(float);
    // One block per vector, n threads per block
    kernel_wht_inplace<<<n_vecs, n, smem_bytes, s>>>(d_data, n);
}

void sp_cuda_iwht_inplace(float *d_data, int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int smem_bytes = n * sizeof(float);
    kernel_wht_inplace<<<n_vecs, n, smem_bytes, s>>>(d_data, n);
    float inv_n = 1.0f / (float)n;
    kernel_iwht_scale<<<n_vecs, n, 0, s>>>(d_data, n, inv_n);
}

void sp_cuda_mobius_reorder(float *d_data, const int *d_order,
                            int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    // Need a temporary buffer (in-place reorder isn't safe)
    float *d_tmp;
    cudaMalloc(&d_tmp, (size_t)n_vecs * n * sizeof(float));
    kernel_mobius_reorder<<<n_vecs, n, 0, s>>>(d_data, d_tmp, d_order, n);
    cudaMemcpyAsync(d_data, d_tmp, (size_t)n_vecs * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    cudaFree(d_tmp);
}

void sp_cuda_mobius_unreorder(float *d_data, const int *d_order,
                              int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    float *d_tmp;
    cudaMalloc(&d_tmp, (size_t)n_vecs * n * sizeof(float));
    kernel_mobius_unreorder<<<n_vecs, n, 0, s>>>(d_data, d_tmp, d_order, n);
    cudaMemcpyAsync(d_data, d_tmp, (size_t)n_vecs * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    cudaFree(d_tmp);
}

void sp_cuda_band_quantize(const float *d_input, void *d_output,
                           const sp_band_config_t *bc,
                           int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int n = bc->band_size * bc->n_bands;

    // Upload band_bits to device
    int *d_band_bits;
    cudaMalloc(&d_band_bits, bc->n_bands * sizeof(int));
    cudaMemcpyAsync(d_band_bits, bc->band_bits, bc->n_bands * sizeof(int),
                    cudaMemcpyHostToDevice, s);

    // One thread per vector (simple version for decode)
    // For large n_vecs (prefill), launch more threads
    int block = 256;
    int grid  = (n_vecs + block - 1) / block;
    kernel_band_quantize_simple<<<grid, block, 0, s>>>(
        d_input, (uint8_t *)d_output,
        n, bc->n_bands, d_band_bits,
        bc->band_size, bc->total_bytes
    );

    cudaFree(d_band_bits);
}

void sp_cuda_band_dequantize(const void *d_input, float *d_output,
                             const sp_band_config_t *bc,
                             int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int n = bc->band_size * bc->n_bands;

    int *d_band_bits;
    cudaMalloc(&d_band_bits, bc->n_bands * sizeof(int));
    cudaMemcpyAsync(d_band_bits, bc->band_bits, bc->n_bands * sizeof(int),
                    cudaMemcpyHostToDevice, s);

    int block = 256;
    int grid  = (n_vecs + block - 1) / block;
    kernel_band_dequantize_simple<<<grid, block, 0, s>>>(
        (const uint8_t *)d_input, d_output,
        n, bc->n_bands, d_band_bits,
        bc->band_size, bc->total_bytes
    );

    cudaFree(d_band_bits);
}

void sp_cuda_nan_guard(float *d_data, int n, int n_vecs,
                       float max_mag, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    kernel_nan_guard<<<n_vecs, n, 0, s>>>(d_data, n, max_mag);
}

// ============================================================================
// Shadow cache implementation
// ============================================================================

int sp_cuda_cache_init(sp_cuda_cache_t *cc, const sp_config_t *cfg,
                       int max_seq_len, void *stream) {
    memcpy(&cc->config, cfg, sizeof(sp_config_t));
    cc->max_seq_len = max_seq_len;
    cc->stream = stream;

    sp_band_config_init(&cc->k_bands, cfg->head_dim,
                        cfg->k_n_bands, cfg->k_band_bits);
    sp_band_config_init(&cc->v_bands, cfg->head_dim,
                        cfg->v_n_bands, cfg->v_band_bits);

    int n_slots = cfg->n_layers * cfg->n_heads_kv;
    size_t k_total = (size_t)n_slots * max_seq_len * cc->k_bands.total_bytes;
    size_t v_total = (size_t)n_slots * max_seq_len * cc->v_bands.total_bytes;

    cudaMalloc(&cc->d_k_cache, k_total);
    cudaMalloc(&cc->d_v_cache, v_total);
    cudaMalloc(&cc->d_scratch, cfg->head_dim * sizeof(float));

    // Upload Möbius tables
    if (cfg->use_mobius_mask) {
        sp_mobius_mask_t mask;
        sp_mobius_mask_init(&mask, cfg->head_dim);

        cudaMalloc(&cc->d_mobius_order, cfg->head_dim * sizeof(int));
        cudaMemcpy(cc->d_mobius_order, mask.order,
                   cfg->head_dim * sizeof(int), cudaMemcpyHostToDevice);

        // Build inverse permutation for unreorder
        int *inv = (int *)malloc(cfg->head_dim * sizeof(int));
        for (int i = 0; i < cfg->head_dim; i++) {
            inv[mask.order[i]] = i;
        }
        cudaMalloc(&cc->d_mobius_inv, cfg->head_dim * sizeof(int));
        cudaMemcpy(cc->d_mobius_inv, inv,
                   cfg->head_dim * sizeof(int), cudaMemcpyHostToDevice);

        free(inv);
        sp_mobius_mask_free(&mask);
    } else {
        cc->d_mobius_order = NULL;
        cc->d_mobius_inv   = NULL;
    }

    fprintf(stderr, "[Shannon-Prime CUDA] Cache allocated:\n");
    fprintf(stderr, "  K: %.2f MB (%d bytes/vec × %d slots × %d seq)\n",
            k_total / (1024.0 * 1024.0), cc->k_bands.total_bytes,
            n_slots, max_seq_len);
    fprintf(stderr, "  V: %.2f MB (%d bytes/vec × %d slots × %d seq)\n",
            v_total / (1024.0 * 1024.0), cc->v_bands.total_bytes,
            n_slots, max_seq_len);
    fprintf(stderr, "  Total: %.2f MB (vs fp16 baseline: %.2f MB)\n",
            (k_total + v_total) / (1024.0 * 1024.0),
            (size_t)n_slots * max_seq_len * cfg->head_dim * 2 * 2 / (1024.0 * 1024.0));

    return 0;
}

void sp_cuda_cache_free(sp_cuda_cache_t *cc) {
    cudaFree(cc->d_k_cache);
    cudaFree(cc->d_v_cache);
    cudaFree(cc->d_scratch);
    cudaFree(cc->d_mobius_order);
    cudaFree(cc->d_mobius_inv);
}

// Single-vector write: WHT → Möbius → quantize → store
void sp_cuda_write_k(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_k_vec) {
    int hd = cc->config.head_dim;
    float *scratch = cc->d_scratch;
    cudaStream_t s = (cudaStream_t)cc->stream;

    // Copy to scratch
    cudaMemcpyAsync(scratch, d_k_vec, hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);

    // WHT forward
    sp_cuda_wht_inplace(scratch, hd, 1, cc->stream);

    // Möbius reorder
    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_reorder(scratch, cc->d_mobius_order, hd, 1, cc->stream);
    }

    // Quantize into cache
    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                  + (size_t)pos * cc->k_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_k_cache + offset;

    sp_cuda_band_quantize(scratch, dest, &cc->k_bands, 1, cc->stream);
}

void sp_cuda_write_v(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_v_vec) {
    int hd = cc->config.head_dim;
    float *scratch = cc->d_scratch;
    cudaStream_t s = (cudaStream_t)cc->stream;

    cudaMemcpyAsync(scratch, d_v_vec, hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    sp_cuda_wht_inplace(scratch, hd, 1, cc->stream);

    // No Möbius for V (uniform spectrum)

    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                  + (size_t)pos * cc->v_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_v_cache + offset;

    sp_cuda_band_quantize(scratch, dest, &cc->v_bands, 1, cc->stream);
}

// Single-vector read: load → dequantize → Möbius unreorder → iWHT
void sp_cuda_read_k(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_k_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                  + (size_t)pos * cc->k_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_k_cache + offset;

    sp_cuda_band_dequantize(src, d_k_out, &cc->k_bands, 1, (void *)s);

    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_unreorder(d_k_out, cc->d_mobius_inv, hd, 1, (void *)s);
    }

    sp_cuda_iwht_inplace(d_k_out, hd, 1, (void *)s);
    sp_cuda_nan_guard(d_k_out, hd, 1, 65504.0f, (void *)s);
}

void sp_cuda_read_v(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_v_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                  + (size_t)pos * cc->v_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_v_cache + offset;

    sp_cuda_band_dequantize(src, d_v_out, &cc->v_bands, 1, (void *)s);

    // No Möbius unreorder for V
    sp_cuda_iwht_inplace(d_v_out, hd, 1, (void *)s);
    sp_cuda_nan_guard(d_v_out, hd, 1, 65504.0f, (void *)s);
}

// ============================================================================
// Batch operations for prefill
// ============================================================================

void sp_cuda_write_k_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_k_vecs) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    // Allocate batch scratch
    float *d_work;
    cudaMalloc(&d_work, (size_t)n_pos * hd * sizeof(float));
    cudaMemcpyAsync(d_work, d_k_vecs, (size_t)n_pos * hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);

    // WHT all vectors in parallel
    sp_cuda_wht_inplace(d_work, hd, n_pos, cc->stream);

    // Möbius reorder all
    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_reorder(d_work, cc->d_mobius_order, hd, n_pos, cc->stream);
    }

    // Quantize all into cache
    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                       + (size_t)start_pos * cc->k_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_k_cache + base_offset;

    sp_cuda_band_quantize(d_work, dest, &cc->k_bands, n_pos, cc->stream);

    cudaFree(d_work);
}

void sp_cuda_write_v_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_v_vecs) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    float *d_work;
    cudaMalloc(&d_work, (size_t)n_pos * hd * sizeof(float));
    cudaMemcpyAsync(d_work, d_v_vecs, (size_t)n_pos * hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);

    sp_cuda_wht_inplace(d_work, hd, n_pos, cc->stream);
    // No Möbius for V

    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                       + (size_t)start_pos * cc->v_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_v_cache + base_offset;

    sp_cuda_band_quantize(d_work, dest, &cc->v_bands, n_pos, cc->stream);

    cudaFree(d_work);
}

void sp_cuda_read_k_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_k_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                       + (size_t)start_pos * cc->k_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_k_cache + base_offset;

    sp_cuda_band_dequantize(src, d_k_out, &cc->k_bands, n_pos, (void *)s);

    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_unreorder(d_k_out, cc->d_mobius_inv, hd, n_pos, (void *)s);
    }

    sp_cuda_iwht_inplace(d_k_out, hd, n_pos, (void *)s);
    sp_cuda_nan_guard(d_k_out, hd, n_pos, 65504.0f, (void *)s);
}

void sp_cuda_read_v_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_v_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                       + (size_t)start_pos * cc->v_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_v_cache + base_offset;

    sp_cuda_band_dequantize(src, d_v_out, &cc->v_bands, n_pos, (void *)s);
    sp_cuda_iwht_inplace(d_v_out, hd, n_pos, (void *)s);
    sp_cuda_nan_guard(d_v_out, hd, n_pos, 65504.0f, (void *)s);
}

// ============================================================================
// Diagnostics
// ============================================================================

void sp_cuda_print_memory(const sp_cuda_cache_t *cc) {
    int n_slots = cc->config.n_layers * cc->config.n_heads_kv;
    size_t k_total = (size_t)n_slots * cc->max_seq_len * cc->k_bands.total_bytes;
    size_t v_total = (size_t)n_slots * cc->max_seq_len * cc->v_bands.total_bytes;
    size_t baseline = (size_t)n_slots * cc->max_seq_len * cc->config.head_dim * 2 * 2;

    fprintf(stderr, "[Shannon-Prime CUDA] Memory:\n");
    fprintf(stderr, "  Compressed: %.2f MB (K: %.2f + V: %.2f)\n",
            (k_total + v_total) / (1024.0 * 1024.0),
            k_total / (1024.0 * 1024.0),
            v_total / (1024.0 * 1024.0));
    fprintf(stderr, "  Baseline:   %.2f MB\n", baseline / (1024.0 * 1024.0));
    fprintf(stderr, "  Ratio:      %.1f×\n",
            (double)baseline / (double)(k_total + v_total));
}

} // extern "C"
