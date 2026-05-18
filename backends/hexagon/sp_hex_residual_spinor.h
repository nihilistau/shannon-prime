// Shannon-Prime — Residual quantize + SU(2) spinor phase (Strike 12).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Takes (actual, predicted) fp32 vectors of `n_lanes` length (currently 140
// for the head_dim=154 mixed-radix config), computes the residual
// (actual - predicted), and packs into:
//
//   [ 53 bytes ] 140 × 3-bit magnitude quantized against per-block amax
//   [ 18 bytes ] 140 × 1-bit SU(2) spinor phase (sign of residual)
//                ---------
//   [ 71 bytes ] per residual block
//
// Plus one fp32 amax scalar returned out-of-band for the downstream decoder.
//
// Compression vs the input: 140 × 4 = 560 bytes raw fp32 → 71 bytes packed
// (+4 bytes amax) = 7.5× on the residual.  Combined with the 14 fp16 skeleton
// coefficients (28 bytes), the full Hierarchical Spinor block is
//   28 + 71 + 4 = 103 bytes
// vs 154 × 4 = 616 bytes uncompressed = **5.98× total compression**.
//
// The HVX kernel runs vsub + vmax-tree-reduce on five 32-lane HVX vectors
// (padded from 140 to 160 — Strike 11b's convention).  Per-lane Q3+phase
// pack is a scalar tail (140 iterations × ~10 cycles ≪ FastRPC overhead).

#ifndef SP_HEX_RESIDUAL_SPINOR_H
#define SP_HEX_RESIDUAL_SPINOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_HEX_RESIDUAL_LANES_MAX  140
#define SP_HEX_RESIDUAL_PAD        160   // ceil(140/32)*32 for HVX
#define SP_HEX_RESIDUAL_MAG_BYTES   53   // ceil(140*3/8)
#define SP_HEX_RESIDUAL_PHASE_BYTES 18   // ceil(140/8)
#define SP_HEX_RESIDUAL_TOTAL_BYTES 71   // mag + phase

// ----------------------------------------------------------------------------
// HVX kernel (DSP side).  Inputs and `packed` must be 128-B aligned.
// `actual` and `predicted` are each `n_padded` fp32 (160 for the canonical
// 140-lane config; tail lanes beyond n_lanes should be zero so they don't
// inflate the amax).
//
// Returns 0 on success, -1 on shape mismatch.
//
// On success, *amax_out holds the per-block fp32 amax (used by the decoder
// to recover the magnitudes).  When all residuals are zero, amax_out = 0
// and packed[0..71) = 0.
// ----------------------------------------------------------------------------
int sp_hex_residual_spinor_hvx(const float *actual,
                                const float *predicted,
                                int n_lanes,
                                int n_padded,
                                uint8_t *packed,
                                int packed_capacity,
                                float *amax_out);

// ----------------------------------------------------------------------------
// Host-side scalar reference.  IDENTICAL math (fp32 throughout, same dense-
// pack semantic).  Bit-equal expected vs the HVX kernel for any (actual,
// predicted) pair — the only fp arithmetic is vsub + vmax + one scalar mul
// per lane, all of which are deterministic between HVX scalar units and
// host fp32 (single-precision IEEE 754).
// ----------------------------------------------------------------------------
int sp_hex_residual_spinor_ref(const float *actual,
                                const float *predicted,
                                int n_lanes,
                                int n_padded,
                                uint8_t *packed,
                                int packed_capacity,
                                float *amax_out);

#ifdef __cplusplus
}
#endif

#endif  // SP_HEX_RESIDUAL_SPINOR_H
