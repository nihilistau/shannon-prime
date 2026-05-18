/* sp_ntt_crt_hvx.h — Hexagon V69 HVX-accelerated CRT NTT kernel.
 *
 * Strike 1 (Path A): port the proven AVX-512 Barrett-reduction kernel
 * to Hexagon V69 HVX intrinsics. Same dual-Proth primes (q1, q2) as the
 * desktop AVX-512 path — output is bit-identical to sp_ntt_crt.c through
 * the shared scalar reference (mulmod_barrett), so the cross-arch
 * parity test is the existing test_sp_ntt_crt.cpp expanded with HVX
 * lanes.
 *
 * Lane shape: HVX 1024-bit vectors = 32 × uint32 per instruction. The
 * 30-bit Proth residues live natively in 32-bit lanes; the 60-bit
 * Barrett intermediates use HVX_VectorPair (2 × 1024 bits = 32 × uint64).
 * 4× lane density vs AVX-512's 8 × uint64.
 *
 * Build gating: under SP_HEXAGON_ENABLED the functions call native
 * HVX intrinsics from hexagon_protos.h. Without the flag (x86 / CI),
 * they fall back to a scalar reference that produces bit-identical
 * output — so the parity test runs on every host.
 *
 * Math invariants preserved (same as sp_ntt_crt.c):
 *   - Negacyclic ring Z_q[x]/(x^N + 1), N = 256
 *   - Pre-twist a_i *= psi^i mod q
 *   - CT-DIT butterflies in omega = psi^2
 *   - Post-scale by N^-1 mod q on inverse
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#ifndef SP_NTT_CRT_HVX_H
#define SP_NTT_CRT_HVX_H

#include <stdint.h>
#include "sp_ntt_crt.h"  /* re-uses sp_ntt_ctx + sp_ntt_crt_combine */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Public API — mirrors sp_ntt_crt.h but routes through HVX ─────── */

/* In-place forward negacyclic NTT under the given prime context.
 *   - input:  signed int64 polynomial coefficients (length SP_NTT_CRT_N)
 *   - output: uint64 NTT-domain values in [0, ctx->q)
 *
 * The HVX path packs the uint64 input into uint32 lanes after the
 * standard `> q` reduction (all residues fit in 30 bits), runs the
 * butterfly + pre-twist in the 32-lane domain, and writes back uint64.
 */
void sp_ntt_crt_hvx_forward(uint64_t* a, const sp_ntt_ctx* ctx);

/* In-place inverse negacyclic NTT — mirror of forward. */
void sp_ntt_crt_hvx_inverse(uint64_t* a, const sp_ntt_ctx* ctx);

/* Pointwise multiply mod ctx->q. c[i] = a[i] * b[i] mod q.
 * The dominant hot-path call in attention (one per (Q,K) pair). */
void sp_ntt_crt_hvx_pointwise_mul(uint64_t* c,
                                  const uint64_t* a,
                                  const uint64_t* b,
                                  const sp_ntt_ctx* ctx);

/* ─── Phase-13/Phase-Engine integration entry points ───────────────── */

/* CRT-stitched dot product (HVX variant of sp_poly_dot_product_ntt_crt_qk_cached).
 * Same signature as the scalar path so callers can swap by symbol. */
float sp_poly_dot_product_ntt_crt_qk_cached_hvx(
        const uint64_t* Q_ntt_q1,
        const uint64_t* Q_ntt_q2,
        const uint64_t* K_ntt_q1,
        const uint64_t* K_ntt_q2,
        int d, double delta,
        uint64_t* c_q1_scratch,
        uint64_t* c_q2_scratch,
        int* ok);

/* ─── HVX context selection ────────────────────────────────────────── */

/* Returns 1 if this build was compiled with SP_HEXAGON_ENABLED and the
 * HVX intrinsics are linked. The forward/inverse/pointwise functions
 * are always callable — they fall back to scalar reference when this
 * returns 0 — so this is informational only (used by sp_attention.cpp
 * to log which path is active). */
int sp_ntt_crt_hvx_available(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SP_NTT_CRT_HVX_H */
