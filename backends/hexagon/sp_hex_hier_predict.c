// Shannon-Prime — Hierarchical Spinor predictor (Strike 11).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// W-matrix predictor running int32 Q15 MAC on V69 HVX.  Reuses the
// validated Strike-5 idiom (Q6_Vw_vmpyieacc_VwVwVh + Q6_V_vsplat_R)
// and adds zero-pad to a multiple of 32 lanes so the kernel processes
// 32 W values per MAC call with full lane utilisation.

#include "sp_hex_hier_predict.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// HVX intrinsics are only available when compiled for the DSP toolchain.
// The host-side reference functions stay pure C so they build on any
// platform (used by parity tests).
#if defined(__hexagon__)
#  include <hexagon_protos.h>
#  include <hexagon_types.h>
#  define SP_HEX_HVX_AVAILABLE 1
#else
#  define SP_HEX_HVX_AVAILABLE 0
#endif

// ----------------------------------------------------------------------------
// Compile-time shape limits.  The kernel currently only handles head_dim=154
// (skeleton=14, predicted=140, padded=160 = 5*32).  Other configs are guarded
// by the dispatch stub in sp_hex_imp.c — the kernel asserts and returns -1.
// ----------------------------------------------------------------------------
#define SP_HEX_HIER_SKELETON_MAX   14
#define SP_HEX_HIER_PREDICTED_MAX  140
#define SP_HEX_HIER_PREDICTED_PAD  160   // ceil(140 / 32) * 32

// ----------------------------------------------------------------------------
// Q15 conversion helpers — fp32 -> int16, saturating round half-up.
// Used by both the HVX kernel (to convert the skeleton scalar before
// broadcast) and the scalar reference.
// ----------------------------------------------------------------------------
static int32_t sp_hex_fp32_to_q15(float v) {
    if (v >  1.0f) v =  1.0f;
    if (v < -1.0f) v = -1.0f;
    float scaled = v * 32767.0f;
    int32_t q = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (q >  32767) q =  32767;
    if (q < -32768) q = -32768;
    return q;
}

static float sp_hex_q30_to_fp32(int32_t acc) {
    // 32767 * 32767 = 1073676289 (just under 2^30); single-precision
    // recip is fine — this is dequantization, not the MAC itself.
    static const float DEQUANT = 1.0f / (32767.0f * 32767.0f);
    return (float)acc * DEQUANT;
}

// ============================================================================
// HVX kernel (DSP only) — int32 vmpyieacc inner loop.
// ============================================================================
//
// Algorithm:
//   acc[i]    = 0                                       (i in 0..160)
//   for j in 0..14:
//     s_j     = round(skeleton[j] * 32767)              (int32 scalar)
//     for c in 0..5:                                    (5 HVX vectors)
//       v_w   = vmem load 32 i32 from W[j, c*32 .. c*32+32]
//       v_acc = vmpyieacc(v_acc, vsplat(s_j), v_w)
//   predicted[i] = acc[i] / (32767 * 32767)             (i in 0..140)
//
// vmpyieacc semantics (Strike-5-validated):
//   Vacc.w[i] += Vu.w[i] * Vv.h[2i]
// With v_w pre-padded so each i32 lane has its Q15 value in the low (even)
// halfword, Vv.h[2i] picks W[i] for each i. 32 MACs per call, fully packed.
// ============================================================================

int sp_hex_hier_predict_hvx_q15(const float *skeleton,
                                 int skeleton_size,
                                 const int32_t *w_matrix_q15_colmajor,
                                 int predicted_size,
                                 float *predicted) {
    if (!skeleton || !w_matrix_q15_colmajor || !predicted) return -1;
    if (skeleton_size != SP_HEX_HIER_SKELETON_MAX ||
        predicted_size != SP_HEX_HIER_PREDICTED_MAX) {
        return -1;
    }

    // Step 1: skeleton fp32 -> Q15 int32 scalars.
    int32_t skel_q15[SP_HEX_HIER_SKELETON_MAX];
    for (int j = 0; j < skeleton_size; ++j) {
        skel_q15[j] = sp_hex_fp32_to_q15(skeleton[j]);
    }

    // Step 2: accumulator buffer (128-byte aligned for vmem stores).
    int32_t acc[SP_HEX_HIER_PREDICTED_PAD] __attribute__((aligned(128)));
    memset(acc, 0, sizeof(acc));

#if SP_HEX_HVX_AVAILABLE
    // ---- HVX path ----
    HVX_Vector *acc_v = (HVX_Vector *)acc;
    const int n_vectors = SP_HEX_HIER_PREDICTED_PAD / 32;  // 5

    for (int j = 0; j < skeleton_size; ++j) {
        const HVX_Vector *w_col_v = (const HVX_Vector *)(
            w_matrix_q15_colmajor + (size_t)j * SP_HEX_HIER_PREDICTED_PAD);
        HVX_Vector v_scal = Q6_V_vsplat_R(skel_q15[j]);
        for (int c = 0; c < n_vectors; ++c) {
            acc_v[c] = Q6_Vw_vmpyieacc_VwVwVh(acc_v[c], v_scal, w_col_v[c]);
        }
    }
#else
    // ---- Fallback scalar path (host build / off-target tests) ----
    // Exact same Q15 math, lane-by-lane.  Inside the loop we read the
    // low halfword of the packed int32 lane to recover the W value.
    for (int j = 0; j < skeleton_size; ++j) {
        const int32_t *w_col =
            w_matrix_q15_colmajor + (size_t)j * SP_HEX_HIER_PREDICTED_PAD;
        int32_t s = skel_q15[j];
        for (int i = 0; i < SP_HEX_HIER_PREDICTED_PAD; ++i) {
            int32_t w_low = (int32_t)(int16_t)(w_col[i] & 0xFFFF);
            acc[i] += s * w_low;
        }
    }
#endif

    // Step 3: dequantize i32 Q30 -> fp32 for the first predicted_size lanes.
    for (int i = 0; i < predicted_size; ++i) {
        predicted[i] = sp_hex_q30_to_fp32(acc[i]);
    }
    return 0;
}

// ============================================================================
// Host-side scalar reference — IDENTICAL Q15 math to the HVX kernel.
// ============================================================================
//
// Bit-equal expected vs the HVX kernel for ANY (skeleton, W) input.  This
// isolates "is the kernel implementing the math correctly" from "is Q15
// a tight enough quantization."
// ============================================================================

int sp_hex_hier_predict_ref_q15(const float *skeleton,
                                 int skeleton_size,
                                 const int32_t *w_matrix_q15_colmajor,
                                 int predicted_size,
                                 int predicted_padded,
                                 float *predicted) {
    if (!skeleton || !w_matrix_q15_colmajor || !predicted) return -1;
    if (skeleton_size <= 0 || predicted_size <= 0) return -1;
    if (predicted_padded < predicted_size) return -1;
    if (predicted_padded % 32 != 0) return -1;

    int32_t *acc = (int32_t *)calloc((size_t)predicted_padded, sizeof(int32_t));
    if (!acc) return -1;

    for (int j = 0; j < skeleton_size; ++j) {
        int32_t s = sp_hex_fp32_to_q15(skeleton[j]);
        const int32_t *w_col = w_matrix_q15_colmajor +
                                (size_t)j * (size_t)predicted_padded;
        for (int i = 0; i < predicted_padded; ++i) {
            // Mirror the HVX vmpyieacc semantic: take the SIGNED low halfword
            // of each i32 lane, sign-extend to i32, then scalar * i32 -> i32
            // (no saturation).  Standard C 2's-complement wraparound is the
            // same as what the HVX MAC produces; total per-position sum has
            // 14 terms, max absolute < 14 * 32767^2 << 2^31 by orders of
            // magnitude (W values are ~5%-scale per the calibration source).
            int32_t w_low = (int32_t)(int16_t)(w_col[i] & 0xFFFF);
            acc[i] += s * w_low;
        }
    }

    for (int i = 0; i < predicted_size; ++i) {
        predicted[i] = sp_hex_q30_to_fp32(acc[i]);
    }
    free(acc);
    return 0;
}

// ============================================================================
// Host-side fp32 reference — no quantization anywhere.  Used to measure the
// Q15 round-trip error budget.  Takes the int16 (unpacked) W matrix.
// ============================================================================

int sp_hex_hier_predict_ref_f32(const float *skeleton,
                                 int skeleton_size,
                                 const int16_t *w_matrix_q15_i16_colmajor,
                                 int predicted_size,
                                 float *predicted) {
    if (!skeleton || !w_matrix_q15_i16_colmajor || !predicted) return -1;
    if (skeleton_size <= 0 || predicted_size <= 0) return -1;
    static const float DEQUANT_W = 1.0f / 32767.0f;

    for (int i = 0; i < predicted_size; ++i) {
        double sum = 0.0;
        for (int j = 0; j < skeleton_size; ++j) {
            float w_f = (float)w_matrix_q15_i16_colmajor[
                            (size_t)j * (size_t)predicted_size + (size_t)i]
                        * DEQUANT_W;
            sum += (double)w_f * (double)skeleton[j];
        }
        predicted[i] = (float)sum;
    }
    return 0;
}

