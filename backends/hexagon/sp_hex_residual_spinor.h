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

// Strike 11c: shape realignment to the engine's actual knight_mask geometry.
// At pad_dim=154 (head_dim=128 sqfree-padded to 2*7*11), the engine's
// sp_knight_mask produces sk_k=14 skeleton + n_res=60 non-squarefree residual.
// The remaining ~80 squarefree-but-not-skeleton indices are dropped (assumed
// zero / reconstructed implicitly by Kronecker structure) — that's the
// VHT2 design that hits the validated PPL 7.32 @ 3.3x on Qwen3-8B Q8.
//
// Lane / padding / pack-byte counts redo:
//   60 lanes -> pad 64 (2 HVX vectors of 32 i32) for the W-matrix MAC kernel
//   60 * 3 bits packed magnitudes = 180 bits = 23 bytes (ceil)
//   60 * 1 bit  packed phases     =  60 bits =  8 bytes (ceil)
//   total packed residual block   = 31 bytes (was 71 at the 140-lane design)
#define SP_HEX_RESIDUAL_LANES_MAX   60
#define SP_HEX_RESIDUAL_PAD         64   // ceil(60/32)*32
#define SP_HEX_RESIDUAL_MAG_BYTES   23   // ceil(60*3/8)
#define SP_HEX_RESIDUAL_PHASE_BYTES  8   // ceil(60/8)
#define SP_HEX_RESIDUAL_TOTAL_BYTES 31   // mag + phase

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

// ----------------------------------------------------------------------------
// Strike 14 — INVERSE pack: take the 71-byte packed blob and amax, expand
// back to `n_lanes` fp32 residual values.  The remaining `n_padded - n_lanes`
// tail lanes are zeroed so a downstream HVX vadd doesn't touch garbage.
//
//   residual[i] = (phase[i] ? -1 : +1) * (mag[i] / 7.0f) * amax
//
// Bit-equal expected between HVX and ref variants — the math is pure scalar
// arithmetic and one scalar fp multiply per lane.  HVX variant is kept as a
// separate entry only so callers can pin it on the DSP TU without pulling
// the host reference's calloc into the freestanding skel build.
//
// Returns 0 on success, -1 on shape mismatch.
// ----------------------------------------------------------------------------
int sp_hex_residual_spinor_unpack_hvx(const uint8_t *packed,
                                       int packed_len,
                                       int n_lanes,
                                       int n_padded,
                                       float amax,
                                       float *residual_out);

int sp_hex_residual_spinor_unpack_ref(const uint8_t *packed,
                                       int packed_len,
                                       int n_lanes,
                                       int n_padded,
                                       float amax,
                                       float *residual_out);

// ----------------------------------------------------------------------------
// Strike 14 combine — `out[i] = a[i] + b[i]` over `n_padded` fp32 lanes.
//
// HVX variant uses Q6_Vqf32_vadd_VsfVsf + Q6_Vsf_equals_Vqf32 over
// `n_padded/32` HVX vectors.  Buffers must be 128-byte aligned.  Off-target
// builds fall through to a scalar add for unit-test parity.
//
// Returns 0 on success, -1 on shape mismatch.
// ----------------------------------------------------------------------------
int sp_hex_residual_combine_hvx(const float *a,
                                 const float *b,
                                 int n_padded,
                                 float *out);

#ifdef __cplusplus
}
#endif

#endif  // SP_HEX_RESIDUAL_SPINOR_H
