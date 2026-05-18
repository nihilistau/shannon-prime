// Shannon-Prime — Hexagon sqfree cache backend (Strike 15a).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Engine integration of the Hierarchical Spinor write/read pipeline.  Mirrors
// the surface of sp_sqfree_cache_t / sp_cuda_sqfree_cache_t exactly so the
// KvCache::Impl branch logic in src/kv_cache.cpp slots it in alongside the
// existing host-scalar and CUDA arms.
//
// Internally:
//   - Pad fp32 K/V (head_dim → pad_dim, host scalar via sp_sqfree_pad_f32)
//   - VHT2 forward (host scalar via sp_vilenkin_forward_f32 — temporary,
//     Strike 13 will fuse this onto the DSP)
//   - Skeleton + residual extraction via the runtime-calibrated knight_mask
//   - FastRPC dispatch: sp_hex_hier_predict_f32 + sp_hex_residual_quantize_spinor
//     (encode), sp_hex_hier_decode_f32 (decode)
//   - Per-slot 103-byte storage: [skel fp16: 28 B][packed: 71 B][amax: 4 B]
//
// Calibration delegates to the engine's existing variance + SVD entropy
// accumulators (host-side), then builds the knight_mask.  Until a
// `hier_set_w_matrix` IDL ships (Strike 15b), the DSP-side W matrix stays
// at the compile-time placeholder in sp_hex_w_matrix_hd154.h — PPL on real
// models will be worse than the calibrated-W ceiling but the lifecycle/
// API surface is exercised end-to-end.

#ifndef SP_HEX_SQFREE_CACHE_H
#define SP_HEX_SQFREE_CACHE_H

#include "../../core/shannon_prime.h"   // sp_config_t, sp_knight_mask_t,
                                         // sp_vilenkin_basis_t, etc.
#include "shannon_prime_hexagon.h"       // sp_hexagon_ctx_t

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Per-slot encoded block layout (103 bytes):
//   bytes [ 0..27]  — 14 fp16 skeleton coefficients
//   bytes [28..98]  — 71 packed residual (140 × 3-bit mag + 140 × 1-bit phase)
//   bytes [99..102] — fp32 amax used by the decoder to recover magnitudes
// ----------------------------------------------------------------------------
#define SP_HEX_SQFREE_SKEL_BYTES   28   // 14 * sizeof(fp16)
#define SP_HEX_SQFREE_PACK_BYTES   71   // matches SP_HEX_RESIDUAL_TOTAL_BYTES
#define SP_HEX_SQFREE_AMAX_BYTES    4   // fp32
#define SP_HEX_SQFREE_SLOT_BYTES  103   // total per (layer, head, pos)

typedef struct {
    sp_config_t          config;
    int                  pad_dim;        // sqfree-padded head_dim (154 for 128)
    int                  residual_bits;  // unused at the DSP level (always 3),
                                         // kept for ABI symmetry with sp_sqfree
    bool                 use_spinor;     // 1-bit phase — always true here

    // Runtime-built skeleton/residual partition (calibrated host-side).
    sp_knight_mask_t     mask;
    sp_vilenkin_basis_t  vilenkin;

    // FastRPC handle to the cDSP session.  Owned externally by the caller via
    // sp_hexagon_init() — we just borrow it.  This lets one ctx serve multiple
    // KvCache instances (e.g., when the engine layers two caches for ablation).
    sp_hexagon_ctx_t    *ctx;
    bool                 owns_ctx;       // true if init allocated it

    // Per-slot 103-byte storage. n_slots = n_layers * n_heads_kv.
    uint8_t            **k_cache;
    uint8_t            **v_cache;
    int                  n_slots;
    int                  max_seq_len;

    // Host scratch (per-call, single-threaded — kv_cache.cpp serialises).
    float               *pad_scratch;    // pad_dim floats
    float               *coeff_scratch;  // pad_dim floats — VHT2 in-place
    float               *residual_pad;   // 160 fp32 (140 valid + 20 zero pad)
    float               *predicted_pad;  // 160 fp32
    float               *recon_pad;      // 160 fp32 (decode output)

    // ── Calibration state (transient, freed after calibrate_end) ─────
    bool                 calibrating;
    double              *calib_sum;
    double              *calib_sum2;
    double              *calib_cov;
    int                  calib_n;
    bool                 use_svd_entropy;
} sp_hex_sqfree_cache_t;

// ----------------------------------------------------------------------------
// init / free.  Mirrors sp_sqfree_cache_init's surface.  Optional `ctx` may be
// passed to share a FastRPC session across caches; pass NULL to allocate one.
// ----------------------------------------------------------------------------
int  sp_hex_sqfree_cache_init(sp_hex_sqfree_cache_t *sc,
                               const sp_config_t *cfg,
                               int max_seq_len,
                               int residual_bits,
                               bool use_spinor,
                               sp_hexagon_ctx_t *shared_ctx /* may be NULL */);
void sp_hex_sqfree_cache_free(sp_hex_sqfree_cache_t *sc);

// ----------------------------------------------------------------------------
// Calibration loop — same pattern as sp_sqfree:
//   begin() → for each warmup vec: feed(vec) → end() rebuilds mask + bands.
// ----------------------------------------------------------------------------
int  sp_hex_sqfree_calibrate_begin(sp_hex_sqfree_cache_t *sc);
void sp_hex_sqfree_calibrate_feed (sp_hex_sqfree_cache_t *sc, const float *vec);
int  sp_hex_sqfree_calibrate_end  (sp_hex_sqfree_cache_t *sc);

// ----------------------------------------------------------------------------
// Write / read — identical signatures to sp_sqfree_write_*/read_*.
// ----------------------------------------------------------------------------
void sp_hex_sqfree_write_k(sp_hex_sqfree_cache_t *sc,
                            int layer, int head, int pos,
                            const float *k_vec);
void sp_hex_sqfree_write_v(sp_hex_sqfree_cache_t *sc,
                            int layer, int head, int pos,
                            const float *v_vec);
void sp_hex_sqfree_read_k (const sp_hex_sqfree_cache_t *sc,
                            int layer, int head, int pos,
                            float *k_out);
void sp_hex_sqfree_read_v (const sp_hex_sqfree_cache_t *sc,
                            int layer, int head, int pos,
                            float *v_out);

#ifdef __cplusplus
}
#endif

#endif  // SP_HEX_SQFREE_CACHE_H
