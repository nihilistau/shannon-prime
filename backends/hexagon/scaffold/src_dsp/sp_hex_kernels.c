// Shannon-Prime VHT2 - Hexagon DSP scaffold (scalar reference kernels).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// This file is the scalar-reference DSP-side math. It compiles with
// hexagon-clang (toolv87) as plain C99 — no HVX intrinsics, no platform-
// specific headers — so we can prove the FastRPC pipeline end-to-end before
// the perf work begins.
//
// HVX REPLACEMENT PLAN:
//   sp_hex_vht2_f32           -> add sp_hex_vht2_f32_hvx using hvx::vector
//                                fp32 ops + the HVX shuffle network.
//   sp_hex_band_quantize      -> int8 SIMD: HVX mul-add + saturate-cast.
//   sp_hex_band_dequantize    -> int8 → fp32 widen/scale, vector at a time.
//
// The HVX kernels go in a separate .c (sp_hex_kernels_hvx.c) gated on
// #ifdef __HEXAGON_HVX__ so the scalar fallback always works for unit tests.

#include "sp_hex_kernels.h"

// ============================================================================
// VHT2 forward (scalar reference).
//
// VHT2 is the Vilenkin-Hartley Transform, basis-2 variant. Self-inverse,
// orthonormal up to 1/sqrt(N). Implemented as a Hadamard-like butterfly:
// log2(N) passes of pairwise (a+b, a-b) at increasing stride.
//
// This matches sp_vht2_forward_f32 in the SP math core (core/shannon_prime.c).
// The version here is intentionally identical so the round-trip smoke test
// is bit-comparable across CPU and DSP.
// ============================================================================

#include <math.h>

void sp_hex_vht2_f32(float *data, int n) {
    if (!data || n < 2) return;

    // Power-of-2 butterfly. Stride starts at 1 and doubles each pass.
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int base = 0; base < n; base += (stride << 1)) {
            for (int i = 0; i < stride; ++i) {
                float a = data[base + i];
                float b = data[base + i + stride];
                data[base + i]          = a + b;
                data[base + i + stride] = a - b;
            }
        }
    }

    // Orthonormal scale (matches the math core).
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; ++i) {
        data[i] *= inv_sqrt_n;
    }
}

// ============================================================================
// Band quantize / dequantize (SCAFFOLD STUBS).
//
// These are placeholders that signal "not implemented" so the FastRPC pipeline
// can build and the round_trip path can exercise just VHT2. Wire to the SP
// math core (or reimplement on DSP) when band-IO smoke tests are needed.
// ============================================================================

int sp_hex_band_quantize_scalar(const float *coeffs, int head_dim,
                                 unsigned char *out, int out_capacity,
                                 int *out_len) {
    (void)coeffs; (void)head_dim; (void)out; (void)out_capacity; (void)out_len;
    return -1;
}

int sp_hex_band_dequantize_scalar(const unsigned char *in, int in_len,
                                   int head_dim, int max_bands,
                                   float *out_coeffs) {
    (void)in; (void)in_len; (void)head_dim; (void)max_bands; (void)out_coeffs;
    return -1;
}
