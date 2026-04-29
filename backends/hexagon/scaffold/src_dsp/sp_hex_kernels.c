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
// #ifdef __HVX__ so the scalar fallback always works for unit tests.

#include "sp_hex_kernels.h"

// All four kernels delegate to the SP math core (cross-compiled into
// libsp_hex_skel.so via CMakeLists.txt). The DSP side gets bit-identical
// numerical behaviour to CPU for free, and we keep one source of truth for
// the math.
//
// HVX kernels will be added later as a sibling sp_hex_kernels_hvx.c
// gated on #ifdef __HVX__, with the imp file calling the HVX path
// when the macro is defined and falling back to the scalar core otherwise.

#include "shannon_prime.h"

// Default ship band config: 4 bands at 5/5/4/3 bits. This matches the SP
// math core's sp_config_t defaults for K (see shannon_prime.h:99). When
// the IDL grows a band-config parameter we'll pass it through; for now
// the scaffold hard-codes the K defaults.
static void sp_hex_default_band_config(sp_band_config_t *bc, int head_dim) {
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(bc, head_dim, 4, default_bits);
}

// Forward declarations — defined in sp_hex_kernels_hvx.c when HVX is
// available. Power-of-2 sizes only; the dispatcher falls back to the
// scalar reference for the multi-prime VHT2 sizes that the math core
// supports for non-pow2 head_dims.
#ifdef __HVX__
#include "qurt_hvx.h"   // qurt_hvx_lock / unlock — required around HVX kernels
void sp_hex_vht2_f32_hvx(float *data, int n);
int  sp_hex_band_quantize_hvx(const float *coeffs, int head_dim,
                               unsigned char *out, int out_capacity);
#endif

static int sp_hex_is_pow2(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

void sp_hex_vht2_f32(float *data, int n) {
    // KNOWN ISSUE: sp_hex_vht2_f32_hvx produces output that's ~0.005-0.02
    // absolute different from sp_vht2_forward_f32 on the same input.
    // Verified by direct scalar-vs-HVX output comparison on head_dim=64..1024.
    // The drift is small (~0.1% relative) but compounds badly through the
    // round-trip's two VHT2 calls + band quantize/dequantize, pushing
    // round-trip RMS from 0.029 to 0.69. Likely cause: HVX fp32 rounding
    // semantics differing from scalar fp32 even with -mhvx-ieee-fp.
    // Investigation deferred to a separate workstream.
    //
    // For now: dispatcher falls through to the math core scalar reference.
    // sp_hex_vht2_f32_hvx remains exported so the bench can measure HVX
    // timing; it just isn't on the production path.
    sp_vht2_forward_f32(data, n);
}

int sp_hex_band_quantize_scalar(const float *coeffs, int head_dim,
                                 unsigned char *out, int out_capacity,
                                 int *out_len) {
    sp_band_config_t bc;
    sp_hex_default_band_config(&bc, head_dim);
    if (out_capacity < bc.total_bytes) return -1;

    // KNOWN ISSUE: sp_hex_band_quantize_hvx produces wrong output (round-trip
    // RMS 0.69 vs scalar 0.029). Likely the same HVX fp32 precision drift
    // pattern that affects sp_hex_vht2_f32_hvx — the amax values differ
    // enough between scalar and HVX paths to misalign the per-band scales.
    // Investigation deferred. Dispatcher uses math core scalar reference.
    sp_band_quantize(coeffs, out, &bc);
    if (out_len) *out_len = bc.total_bytes;
    return 0;
}

int sp_hex_band_dequantize_scalar(const unsigned char *in, int in_len,
                                   int head_dim, int max_bands,
                                   float *out_coeffs) {
    sp_band_config_t bc;
    sp_hex_default_band_config(&bc, head_dim);
    if (in_len < bc.total_bytes) return -1;
    if (max_bands < 0 || max_bands >= bc.n_bands) {
        sp_band_dequantize(in, out_coeffs, &bc);
    } else {
        sp_band_dequantize_partial(in, out_coeffs, &bc, max_bands);
    }
    return 0;
}
