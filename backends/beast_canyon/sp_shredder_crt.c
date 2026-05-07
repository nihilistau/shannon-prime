// Shannon-Prime Beast Canyon: Residue-Aware Shredder
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Fused Q8_0 dequant + CRT residue split in a single pass.
// Eliminates the fp32 scratch buffer entirely — fp32 intermediate
// exists only in registers.

#include "sp_shredder_crt.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
static uint64_t sp_shredcrt_time_us(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000ULL / f.QuadPart);
}
#else
#  include <time.h>
static uint64_t sp_shredcrt_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000;
}
#endif

// ---------------------------------------------------------------------------
//  fp16 -> fp32 conversion (software fallback)
// ---------------------------------------------------------------------------
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (man == 0) {
            f = sign; // +/- zero
        } else {
            // Denormal: convert to normal float
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)man << 13); // inf/nan
    } else {
        f = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)man << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
//  CPUID-based AVX-512 detection
// ---------------------------------------------------------------------------
static bool detect_avx512(void) {
#if defined(__GNUC__) || defined(__clang__)
#  if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f");
#  else
    return false;
#  endif
#elif defined(_MSC_VER)
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 16)) != 0;  // EBX bit 16 = AVX-512F
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_init
// ---------------------------------------------------------------------------
int sp_shredder_crt_init(sp_shredder_crt_t* ctx, int expert_dim) {
    if (!ctx || expert_dim <= 0) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->config.expert_dim      = expert_dim;
    ctx->config.use_avx512      = detect_avx512();
    ctx->config.res_buffer_size = (size_t)expert_dim * sizeof(int32_t);

    // Allocate residue buffers — 64-byte aligned for AVX-512.
    size_t alloc_size = (ctx->config.res_buffer_size + 63) & ~(size_t)63;

#ifdef _WIN32
    ctx->res1_buffer = (int32_t*)_aligned_malloc(alloc_size, 64);
    ctx->res2_buffer = (int32_t*)_aligned_malloc(alloc_size, 64);
#else
    ctx->res1_buffer = (int32_t*)aligned_alloc(64, alloc_size);
    ctx->res2_buffer = (int32_t*)aligned_alloc(64, alloc_size);
#endif

    if (!ctx->res1_buffer || !ctx->res2_buffer) {
        sp_shredder_crt_free(ctx);
        return -1;
    }

    memset(ctx->res1_buffer, 0, alloc_size);
    memset(ctx->res2_buffer, 0, alloc_size);

    // TODO: If CUDA is available, upgrade res1_buffer to cudaMallocHost
    // for pinned async copy path. If Intel UHD SVM is available, upgrade
    // res2_buffer to clSVMAlloc for LLC-resident coherent writes.

    return 0;
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_free
// ---------------------------------------------------------------------------
void sp_shredder_crt_free(sp_shredder_crt_t* ctx) {
    if (!ctx) return;
#ifdef _WIN32
    if (ctx->res1_buffer) _aligned_free(ctx->res1_buffer);
    if (ctx->res2_buffer) _aligned_free(ctx->res2_buffer);
#else
    if (ctx->res1_buffer) free(ctx->res1_buffer);
    if (ctx->res2_buffer) free(ctx->res2_buffer);
#endif
    memset(ctx, 0, sizeof(*ctx));
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_q8_row — scalar reference path
// ---------------------------------------------------------------------------
//
// Q8_0 block layout (34 bytes):
//   bytes [0..1]  : fp16 scale (little-endian)
//   bytes [2..33] : 32 × int8 quantised values
//
void sp_shredder_crt_q8_row(sp_shredder_crt_t* ctx,
                            const void* q8_blocks,
                            int32_t* res1_row,
                            int32_t* res2_row,
                            int row_width) {
    if (!ctx || !q8_blocks || !res1_row || !res2_row) return;

    const uint8_t* src = (const uint8_t*)q8_blocks;
    int n_blocks = row_width / SP_SHREDDER_CRT_BLOCK_SIZE;

    // TODO: AVX-512 vectorized path.
    // When ctx->config.use_avx512 is true, the inner loop can process
    // 16 elements per __m512i register:
    //   1. _mm512_cvtepi8_epi32 to widen 16 int8 -> 16 int32
    //   2. _mm512_cvtepi32_ps to convert to fp32
    //   3. _mm512_mul_ps by (block_scale * Q_MAX) broadcast
    //   4. _mm512_cvtps_epi32 with rounding
    //   5. Handle negatives: _mm512_add_epi32 with M1 where mask < 0
    //   6. _mm512_rem_epu32 for M1 and M2 residues
    //   7. _mm512_stream_si512 non-temporal store to res1/res2
    //
    // For now, use the scalar reference path via sp_shredder_crt_scale_element.

    for (int b = 0; b < n_blocks; b++) {
        // Extract fp16 scale from first 2 bytes of block.
        uint16_t scale_fp16;
        memcpy(&scale_fp16, src, 2);
        float block_scale = fp16_to_fp32(scale_fp16);
        src += 2;

        // Process 32 int8 quantised values.
        const int8_t* quants = (const int8_t*)src;
        int base_idx = b * SP_SHREDDER_CRT_BLOCK_SIZE;

        for (int q = 0; q < SP_SHREDDER_CRT_BLOCK_SIZE; q++) {
            float val = (float)quants[q];
            sp_shredder_crt_scale_element(val, block_scale,
                                          &res1_row[base_idx + q],
                                          &res2_row[base_idx + q]);
        }

        src += SP_SHREDDER_CRT_BLOCK_SIZE;
    }
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_q8_expert
// ---------------------------------------------------------------------------
void sp_shredder_crt_q8_expert(sp_shredder_crt_t* ctx,
                               const void* expert_weights,
                               int n_rows,
                               int row_width) {
    if (!ctx || !expert_weights || n_rows <= 0 || row_width <= 0) return;

    uint64_t t0 = sp_shredcrt_time_us();

    const uint8_t* src = (const uint8_t*)expert_weights;
    // Q8_0: each row is (row_width / 32) blocks of 34 bytes each.
    int blocks_per_row = row_width / SP_SHREDDER_CRT_BLOCK_SIZE;
    size_t row_bytes = (size_t)blocks_per_row * (2 + SP_SHREDDER_CRT_BLOCK_SIZE);

    for (int r = 0; r < n_rows; r++) {
        int32_t* r1_row = ctx->res1_buffer + r * row_width;
        int32_t* r2_row = ctx->res2_buffer + r * row_width;

        sp_shredder_crt_q8_row(ctx, src, r1_row, r2_row, row_width);
        src += row_bytes;
    }

    uint64_t t1 = sp_shredcrt_time_us();

    ctx->total_shreds++;
    ctx->total_elements += (uint64_t)n_rows * (uint64_t)row_width;
    ctx->total_us += (t1 - t0);
}

// ---------------------------------------------------------------------------
//  sp_shredder_crt_auto
// ---------------------------------------------------------------------------
int sp_shredder_crt_auto(sp_shredder_crt_t* ctx,
                         uint32_t ggml_type,
                         const void* src,
                         int32_t* res1,
                         int32_t* res2,
                         size_t n_elements) {
    if (!ctx || !src || !res1 || !res2) return -1;

    // GGML_TYPE_Q8_0 = 8
    if (ggml_type == 8) {
        int row_width = ctx->config.expert_dim;
        if (row_width <= 0) return -1;
        int n_rows = (int)(n_elements / (size_t)row_width);
        if (n_rows <= 0) return -1;

        // Temporarily swap in the caller's output buffers.
        int32_t* save_r1 = ctx->res1_buffer;
        int32_t* save_r2 = ctx->res2_buffer;
        ctx->res1_buffer = res1;
        ctx->res2_buffer = res2;

        sp_shredder_crt_q8_expert(ctx, src, n_rows, row_width);

        ctx->res1_buffer = save_r1;
        ctx->res2_buffer = save_r2;
        return 0;
    }

    // TODO: GGML_TYPE_Q4_0 (2), Q4_1 (3), Q5_K_M, Q6_K
    fprintf(stderr, "[sp-shredder-crt] unsupported ggml type %u\n", ggml_type);
    return -1;
}
