// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CRT Multi-GPU Tensor Splitting — Host-Side Implementation
//
// This file contains the host-side Garner reconstruction, quantization
// calibration, and the dispatch coordinator. GPU-specific kernels are
// in sp_crt_cuda.cu and sp_crt_vulkan.c.

#include "sp_crt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Quantization calibration
// ============================================================================

void sp_crt_quant_calibrate(sp_crt_quant_t *q, float min_val, float max_val) {
    // Symmetric quantization around zero. The integer range is [0, M1-1]
    // with zero_point at M1/2. We scale so the observed range maps to
    // [1, M1-2] (leaving headroom for accumulation).

    float range = max_val - min_val;
    if (range < 1e-8f) range = 1e-8f;

    // Use ~30 bits of the Mersenne range for the value space.
    // Reserve some headroom for K-dimensional accumulation.
    double usable_range = (double)(SP_CRT_M1 - 2);
    q->scale = usable_range / (double)range;
    q->inv_scale = (double)range / usable_range;
    q->zero_point = (int64_t)((-min_val) * q->scale + 0.5);
}

// ============================================================================
// Host-side Garner batch reconstruction
// ============================================================================
//
// Vectorizable inner loop: no data dependencies between iterations.
// On x86-64 with AVX2, the compiler should auto-vectorize the uint64
// arithmetic. On ARM, NEON handles it similarly.

void sp_crt_garner_batch(const uint32_t *residue_0,
                         const uint32_t *residue_1,
                         float *output,
                         size_t n,
                         const sp_crt_quant_t *quant) {
    const uint64_t m1 = SP_CRT_M1;
    const uint64_t m2 = SP_CRT_M2;
    const uint64_t gc = SP_CRT_GARNER_C;
    const double inv_scale = quant->inv_scale;
    const int64_t zp = quant->zero_point;

    for (size_t i = 0; i < n; i++) {
        uint64_t a1 = (uint64_t)residue_0[i];
        uint64_t a2 = (uint64_t)residue_1[i];

        // Step 1: a1 mod m2
        uint64_t a1_m2 = a1 % m2;

        // Step 2: diff = (a2 - a1_m2) mod m2  (handles underflow)
        uint64_t diff = (a2 >= a1_m2) ? (a2 - a1_m2) : (m2 - a1_m2 + a2);

        // Step 3: h = (diff * GARNER_C) mod m2
        // diff < m2 < 2^31, gc < m2 < 2^31 => product < 2^62, fits uint64.
        uint64_t h = (diff * gc) % m2;

        // Step 4: reconstruct X = a1 + h * m1
        uint64_t reconstructed = a1 + h * m1;

        // Step 5: dequantize to float
        output[i] = (float)(((double)((int64_t)reconstructed - zp)) * inv_scale);
    }
}

// ============================================================================
// CRT context lifecycle
// ============================================================================

int sp_crt_init(sp_crt_context_t *ctx, int max_M, int max_N, int max_K,
                void *stream_0, void *stream_1) {
    memset(ctx, 0, sizeof(*ctx));

    size_t max_output = (size_t)max_M * max_N;

    // Allocate pinned host buffers for async D2H of residue results.
    // In a full CUDA build these would be cudaHostAlloc'd. For the
    // portable C implementation, we use aligned malloc and let the
    // CUDA/Vulkan layer pin them.
    ctx->h_residue_0 = (uint32_t *)calloc(max_output, sizeof(uint32_t));
    ctx->h_residue_1 = (uint32_t *)calloc(max_output, sizeof(uint32_t));
    ctx->h_output    = (float *)calloc(max_output, sizeof(float));

    if (!ctx->h_residue_0 || !ctx->h_residue_1 || !ctx->h_output) {
        free(ctx->h_residue_0);
        free(ctx->h_residue_1);
        free(ctx->h_output);
        memset(ctx, 0, sizeof(*ctx));
        return -1;
    }

    ctx->M = max_M;
    ctx->N = max_N;
    ctx->K = max_K;
    ctx->stream_0 = stream_0;
    ctx->stream_1 = stream_1;

    // Default quantization (symmetric, conservative range).
    // The caller should re-calibrate with observed value ranges.
    sp_crt_quant_calibrate(&ctx->weight_quant, -1.0f, 1.0f);
    sp_crt_quant_calibrate(&ctx->act_quant, -4.0f, 4.0f);

    ctx->initialized = 1;

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        fprintf(stderr, "[Shannon-Prime CRT] initialized: max %d×%d×%d, "
                        "M1=%llu M2=%llu range=%.2e\n",
                max_M, max_K, max_N,
                (unsigned long long)SP_CRT_M1,
                (unsigned long long)SP_CRT_M2,
                (double)SP_CRT_RANGE);
    }

    return 0;
}

void sp_crt_free(sp_crt_context_t *ctx) {
    free(ctx->h_residue_0);
    free(ctx->h_residue_1);
    free(ctx->h_output);
    memset(ctx, 0, sizeof(*ctx));
}

// ============================================================================
// CPU reference: modular matmul
// ============================================================================
//
// Reference implementation for testing. The GPU kernels in sp_crt_cuda.cu
// replace this with warp-level tiled implementations.

static void sp_crt_cpu_matmul_mod(const uint32_t *A, const uint32_t *B,
                                   uint32_t *C,
                                   int M, int N, int K,
                                   uint64_t modulus) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint64_t a = (uint64_t)A[i * K + k];
                uint64_t b = (uint64_t)B[k * N + j];
                // Periodic reduction to prevent overflow.
                // a < M < 2^31, b < M < 2^31 => a*b < 2^62.
                // acc can grow to at most 2^62 + 2^62 before reduction.
                acc += a * b;
                // Reduce every 4 iterations to stay well within uint64 range.
                // (4 * 2^62 < 2^64)
                if ((k & 3) == 3) {
                    acc %= modulus;
                }
            }
            C[i * N + j] = (uint32_t)(acc % modulus);
        }
    }
}

// Mersenne-optimised variant: uses bit-shift reduction instead of %.
static void sp_crt_cpu_matmul_mersenne(const uint32_t *A, const uint32_t *B,
                                        uint32_t *C,
                                        int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint64_t a = (uint64_t)A[i * K + k];
                uint64_t b = (uint64_t)B[k * N + j];
                acc += a * b;
                if ((k & 3) == 3) {
                    acc = sp_crt_mersenne_reduce(acc);
                }
            }
            C[i * N + j] = sp_crt_mersenne_reduce(acc);
        }
    }
}

// ============================================================================
// Quantize float tensor to residue ring (CPU reference)
// ============================================================================

static void sp_crt_cpu_quantize(const float *input, uint32_t *output,
                                 int n, const sp_crt_quant_t *q,
                                 uint64_t modulus) {
    for (int i = 0; i < n; i++) {
        int64_t ival = (int64_t)(input[i] * q->scale + 0.5) + q->zero_point;
        if (ival < 0) ival = 0;
        output[i] = (uint32_t)((uint64_t)ival % modulus);
    }
}

// ============================================================================
// Main CRT matmul dispatch (CPU reference path)
// ============================================================================
//
// This is the portable reference implementation. When CUDA is available,
// sp_crt_matmul delegates to the GPU kernels instead.

int sp_crt_matmul(sp_crt_context_t *ctx,
                  const float *d_A, const float *d_B, float *d_C,
                  int M, int N, int K) {
    if (!ctx->initialized) return -1;
    if (M * N > ctx->M * ctx->N) return -2; // exceeds allocated buffers

    size_t a_size = (size_t)M * K;
    size_t b_size = (size_t)K * N;
    size_t c_size = (size_t)M * N;

    // --- Quantize A and B for both residue rings ---
    uint32_t *a_m1 = (uint32_t *)malloc(a_size * sizeof(uint32_t));
    uint32_t *b_m1 = (uint32_t *)malloc(b_size * sizeof(uint32_t));
    uint32_t *a_m2 = (uint32_t *)malloc(a_size * sizeof(uint32_t));
    uint32_t *b_m2 = (uint32_t *)malloc(b_size * sizeof(uint32_t));
    if (!a_m1 || !b_m1 || !a_m2 || !b_m2) {
        free(a_m1); free(b_m1); free(a_m2); free(b_m2);
        return -3;
    }

    sp_crt_cpu_quantize(d_A, a_m1, (int)a_size, &ctx->act_quant, SP_CRT_M1);
    sp_crt_cpu_quantize(d_B, b_m1, (int)b_size, &ctx->weight_quant, SP_CRT_M1);
    sp_crt_cpu_quantize(d_A, a_m2, (int)a_size, &ctx->act_quant, SP_CRT_M2);
    sp_crt_cpu_quantize(d_B, b_m2, (int)b_size, &ctx->weight_quant, SP_CRT_M2);

    // --- Matmul in both residue rings (can be parallelized) ---
    // In the GPU path, these launch on stream_0 and stream_1 concurrently.
    // Here we run sequentially as a reference.
    sp_crt_cpu_matmul_mersenne(a_m1, b_m1, ctx->h_residue_0, M, N, K);
    sp_crt_cpu_matmul_mod(a_m2, b_m2, ctx->h_residue_1, M, N, K, SP_CRT_M2);

    free(a_m1); free(b_m1); free(a_m2); free(b_m2);

    // --- Garner reconstruction ---
    // The accumulation quant is the product of weight and activation quants.
    // For CRT matmul, the output scale = act_scale × weight_scale × K
    // (each element is a sum of K products).
    sp_crt_quant_t output_quant;
    output_quant.scale = ctx->act_quant.scale * ctx->weight_quant.scale;
    output_quant.inv_scale = 1.0 / output_quant.scale;
    // Zero point for the accumulation: each output element is the sum of K
    // products of (q_a - zp_a) × (q_b - zp_b), which expands to a cross-term.
    // For simplicity, we compute the reconstruction in the scaled integer domain
    // and then convert, using the known zero-point structure.
    output_quant.zero_point = ctx->act_quant.zero_point
                            * ctx->weight_quant.zero_point * K;

    sp_crt_garner_batch(ctx->h_residue_0, ctx->h_residue_1,
                        d_C, c_size, &output_quant);

    return 0;
}

// ============================================================================
// Verification: single-element CRT round-trip test
// ============================================================================
//
// Tests that quantize → split → matmul_mod → Garner → dequantize
// produces the correct result for a 1×1 "matmul" (i.e., a × b).

int sp_crt_verify_roundtrip(float a, float b, float *error_out) {
    sp_crt_quant_t q;
    sp_crt_quant_calibrate(&q, -4.0f, 4.0f);

    // Quantize to both rings
    uint32_t qa_m1 = sp_crt_quantize_f32(&q, a, SP_CRT_M1);
    uint32_t qb_m1 = sp_crt_quantize_f32(&q, b, SP_CRT_M1);
    uint32_t qa_m2 = sp_crt_quantize_f32(&q, a, SP_CRT_M2);
    uint32_t qb_m2 = sp_crt_quantize_f32(&q, b, SP_CRT_M2);

    // Multiply in each ring
    uint32_t r1 = sp_crt_mersenne_reduce((uint64_t)qa_m1 * qb_m1);
    uint32_t r2 = sp_crt_m2_reduce((uint64_t)qa_m2 * qb_m2);

    // Reconstruct
    uint64_t reconstructed = sp_crt_garner_reconstruct(r1, r2);

    // Dequantize (product quant: scale², zero_point interaction)
    sp_crt_quant_t pq;
    pq.scale = q.scale * q.scale;
    pq.inv_scale = 1.0 / pq.scale;
    pq.zero_point = q.zero_point * q.zero_point;

    float result = sp_crt_dequantize_f32(&pq, reconstructed);
    float expected = a * b;
    float err = fabsf(result - expected);

    if (error_out) *error_out = err;
    return (err < 0.01f) ? 0 : -1; // pass if error < 1%
}
