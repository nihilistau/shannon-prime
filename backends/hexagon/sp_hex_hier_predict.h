// Shannon-Prime — Hierarchical Spinor predictor (Strike 11).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Predicts 140 residual fp32 coefficients from a 14-element Knight skeleton
// via a baked Q15 W-matrix.  HVX inner kernel runs as int32 MAC accumulation
// using the Strike-5-validated Q6_Vw_vmpyieacc_VwVwVh intrinsic family.
//
// W matrix layout (column-major int32, low half = Q15, high half = 0):
//   storage shape : (skeleton_size, predicted_padded)
//   per-lane      : low 16 bits = signed Q15 W; high 16 bits = 0
//   predicted_padded = ceil(predicted_size / 32) * 32
//
// Currently supports head_dim=154 (skeleton=14, predicted=140, padded=160).

#ifndef SP_HEX_HIER_PREDICT_H
#define SP_HEX_HIER_PREDICT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// HVX kernel (DSP side).  W matrix in column-major Q15-padded-int32 lanes.
// Inputs:
//   skeleton                 : fp32, length skeleton_size
//   w_matrix_q15_colmajor    : int32, length skeleton_size * predicted_padded
//                              (caller is responsible for padding to a
//                              multiple of 32; the kernel infers it from
//                              skeleton_size / predicted_size against the
//                              compile-time bank).
// Output:
//   predicted                : fp32, length predicted_size
// Returns 0 on success, -1 on shape mismatch.
//
// The kernel uses int32 MAC chunks of 32 lanes each and produces 5 HVX
// vectors of accumulators for the (140 -> 160-padded) case; the last 20
// lanes are dropped on write-out.
int sp_hex_hier_predict_hvx_q15(const float *skeleton,
                                 int skeleton_size,
                                 const int32_t *w_matrix_q15_colmajor,
                                 int predicted_size,
                                 float *predicted);

// ----------------------------------------------------------------------------
// Host-side scalar reference for parity testing.  Runs the IDENTICAL Q15 math
// to the HVX kernel (no fp32 shortcuts), so a bit-equal compare validates the
// HVX implementation against the Q15 contract.  Use the fp32 reference
// (sp_hex_hier_predict_ref_f32 below) to assess the quantization budget.
int sp_hex_hier_predict_ref_q15(const float *skeleton,
                                 int skeleton_size,
                                 const int32_t *w_matrix_q15_colmajor,
                                 int predicted_size,
                                 int predicted_padded,
                                 float *predicted);

// ----------------------------------------------------------------------------
// Pure fp32 reference (no quantization).  Used to characterize the Q15 error
// budget — DSP output should match this within `2 * max_abs_quant_err` per
// the bake-time number reported by gen_w_matrix.py.
//
// Takes the int16 Q15 W matrix (not padded int32) since this is the "what the
// math actually is" reference; the padding is a kernel implementation detail.
int sp_hex_hier_predict_ref_f32(const float *skeleton,
                                 int skeleton_size,
                                 const int16_t *w_matrix_q15_i16_colmajor,
                                 int predicted_size,
                                 float *predicted);

#ifdef __cplusplus
}
#endif

#endif  // SP_HEX_HIER_PREDICT_H
