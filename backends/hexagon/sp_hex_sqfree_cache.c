// Shannon-Prime — Hexagon sqfree cache backend (Strike 15a impl).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Mirrors sp_sqfree_cache_t lifecycle but routes the heavy math through
// FastRPC to V69 HVX kernels (Strikes 11b + 12 + 14).  ARM-side: pad,
// VHT2 forward, skeleton/residual split, inverse VHT2, unpad.  DSP-side:
// W-matrix predict, residual quantize + spinor pack, decode.

#include "sp_hex_sqfree_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SP_HEXAGON_FASTRPC
#  include "rpcmem.h"
#  include "remote.h"
#  include "AEEStdErr.h"
#  include "sp_hex.h"   // qaic-generated stub header
#  ifndef CDSP_DOMAIN_ID
#    define CDSP_DOMAIN_ID 3
#  endif
#  ifndef CDSP_DOMAIN
#    define CDSP_DOMAIN "&_dom=cdsp"
#  endif
#endif

// Strike 11c: predicted residual lane count = 60 (non-squarefree indices the
// engine's knight_mask classifies as residual), padded to 64 lanes for the
// HVX MAC kernel.  Matches SP_HEX_W_MATRIX_HD154_PREDICTED / _PREDICTED_PAD.
#define SP_HEX_HIER_LANES   60
#define SP_HEX_HIER_PAD     64
#define SP_HEX_HIER_SKEL    14

// Stub-mode fallback when SP_HEXAGON_FASTRPC is not defined — every entry
// point is a no-op so the engine can still link on x86/desktop builds.
#ifndef SP_HEXAGON_FASTRPC

int sp_hex_sqfree_cache_init(sp_hex_sqfree_cache_t *sc, const sp_config_t *cfg,
                              int max_seq_len, int residual_bits,
                              bool use_spinor, sp_hexagon_ctx_t *shared_ctx) {
    (void)sc; (void)cfg; (void)max_seq_len; (void)residual_bits;
    (void)use_spinor; (void)shared_ctx;
    fprintf(stderr, "[sp-engine] sp_hex_sqfree_cache: built without "
                    "SP_HEXAGON_FASTRPC — backend unavailable\n");
    return -1;
}
void sp_hex_sqfree_cache_free(sp_hex_sqfree_cache_t *sc) { (void)sc; }
int  sp_hex_sqfree_calibrate_begin(sp_hex_sqfree_cache_t *sc) { (void)sc; return -1; }
void sp_hex_sqfree_calibrate_feed (sp_hex_sqfree_cache_t *sc, const float *v) { (void)sc; (void)v; }
int  sp_hex_sqfree_calibrate_end  (sp_hex_sqfree_cache_t *sc) { (void)sc; return -1; }
void sp_hex_sqfree_write_k(sp_hex_sqfree_cache_t *sc, int l, int h, int p, const float *v) { (void)sc;(void)l;(void)h;(void)p;(void)v; }
void sp_hex_sqfree_write_v(sp_hex_sqfree_cache_t *sc, int l, int h, int p, const float *v) { (void)sc;(void)l;(void)h;(void)p;(void)v; }
void sp_hex_sqfree_read_k (const sp_hex_sqfree_cache_t *sc, int l, int h, int p, float *o) { (void)sc;(void)l;(void)h;(void)p; if (o) memset(o, 0, sizeof(float) * 128); }
void sp_hex_sqfree_read_v (const sp_hex_sqfree_cache_t *sc, int l, int h, int p, float *o) { (void)sc;(void)l;(void)h;(void)p; if (o) memset(o, 0, sizeof(float) * 128); }

#else // SP_HEXAGON_FASTRPC defined

// ============================================================================
// Internal handle management.  We open our own FastRPC session per cache —
// shared_ctx is reserved for a future Strike that adds an ABI getter on
// sp_hexagon_ctx_t to share sessions across caches.  Today's contract:
// pass shared_ctx = NULL.
// ============================================================================

typedef struct {
    remote_handle64 handle;
    bool            opened;
} sp_hex_sqfree_session_t;

// Stashed in sc->ctx as a void* (we don't own the sp_hexagon_ctx_t — we own a
// session_t wrapper).  Keeping the field shape lets us swap to shared_ctx
// later without breaking the public header.
static sp_hex_sqfree_session_t *sess_from_cache(const sp_hex_sqfree_cache_t *sc) {
    return (sp_hex_sqfree_session_t *)sc->ctx;
}

// ============================================================================
// Lifecycle
// ============================================================================

int sp_hex_sqfree_cache_init(sp_hex_sqfree_cache_t *sc,
                              const sp_config_t *cfg,
                              int max_seq_len,
                              int residual_bits,
                              bool use_spinor,
                              sp_hexagon_ctx_t *shared_ctx) {
    if (!sc || !cfg || max_seq_len <= 0) return -1;
    if (shared_ctx) {
        fprintf(stderr, "[sp_hex_sqfree] shared_ctx not yet supported — "
                        "falling back to dedicated FastRPC session\n");
    }

    memset(sc, 0, sizeof(*sc));
    sc->config        = *cfg;
    sc->pad_dim       = sp_sqfree_pad_dim(cfg->head_dim);
    sc->residual_bits = residual_bits;
    sc->use_spinor    = use_spinor;
    sc->max_seq_len   = max_seq_len;

    // Only pad_dim=154 (head_dim=128) is supported in the W rodata bank today.
    if (sc->pad_dim != 154) {
        fprintf(stderr, "[sp_hex_sqfree] pad_dim=%d unsupported (only 154 "
                        "for head_dim=128 in current W rodata bank)\n", sc->pad_dim);
        return -1;
    }

    // Default knight_mask: sequential skeleton at the first 14 squarefree
    // indices, residual = the remaining 140 indices.  Calibration replaces
    // this with a variance-ranked layout in calibrate_end().
    if (sp_knight_mask_init(&sc->mask, sc->pad_dim, SP_HEX_HIER_SKEL, NULL) != 0) {
        fprintf(stderr, "[sp_hex_sqfree] knight_mask_init failed\n");
        return -1;
    }
    if (sc->mask.sk_k != SP_HEX_HIER_SKEL || sc->mask.n_res != SP_HEX_HIER_LANES) {
        fprintf(stderr, "[sp_hex_sqfree] mask shape mismatch: sk_k=%d n_res=%d "
                        "(expected %d/%d)\n",
                sc->mask.sk_k, sc->mask.n_res,
                SP_HEX_HIER_SKEL, SP_HEX_HIER_LANES);
        sp_knight_mask_free(&sc->mask);
        return -1;
    }

    // Per-slot 103-byte storage.
    sc->n_slots = cfg->n_layers * cfg->n_heads_kv;
    sc->k_cache = (uint8_t **)calloc((size_t)sc->n_slots, sizeof(uint8_t *));
    sc->v_cache = (uint8_t **)calloc((size_t)sc->n_slots, sizeof(uint8_t *));
    if (!sc->k_cache || !sc->v_cache) {
        sp_hex_sqfree_cache_free(sc);
        return -1;
    }
    const size_t slot_bytes = (size_t)max_seq_len * SP_HEX_SQFREE_SLOT_BYTES;
    for (int s = 0; s < sc->n_slots; ++s) {
        sc->k_cache[s] = (uint8_t *)calloc(slot_bytes, 1);
        sc->v_cache[s] = (uint8_t *)calloc(slot_bytes, 1);
        if (!sc->k_cache[s] || !sc->v_cache[s]) {
            sp_hex_sqfree_cache_free(sc);
            return -1;
        }
    }

    // Host scratches (no HVX alignment requirement — never crosses the
    // FastRPC boundary).
    sc->pad_scratch   = (float *)malloc((size_t)sc->pad_dim * sizeof(float));
    sc->coeff_scratch = (float *)malloc((size_t)sc->pad_dim * sizeof(float));
    // rpcmem-backed FastRPC argument buffers (128-byte aligned for HVX).
    sc->residual_pad  = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                               RPCMEM_DEFAULT_FLAGS,
                                               (int)(SP_HEX_HIER_PAD * sizeof(float)));
    sc->predicted_pad = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                               RPCMEM_DEFAULT_FLAGS,
                                               (int)(SP_HEX_HIER_PAD * sizeof(float)));
    sc->recon_pad     = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                               RPCMEM_DEFAULT_FLAGS,
                                               (int)(SP_HEX_HIER_LANES * sizeof(float)));
    if (!sc->pad_scratch || !sc->coeff_scratch ||
        !sc->residual_pad || !sc->predicted_pad || !sc->recon_pad) {
        fprintf(stderr, "[sp_hex_sqfree] scratch alloc failed\n");
        sp_hex_sqfree_cache_free(sc);
        return -1;
    }

    // Open the FastRPC session.  Enable unsigned PD on cDSP — matches the
    // existing sp_hexagon backend and the parity-test scaffold.
    sp_hex_sqfree_session_t *sess =
        (sp_hex_sqfree_session_t *)calloc(1, sizeof(*sess));
    if (!sess) { sp_hex_sqfree_cache_free(sc); return -1; }
    sess->handle = (remote_handle64)-1;
    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        (void)remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     (void *)&data, sizeof(data));
    }
    int rc = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &sess->handle);
    if (rc != AEE_SUCCESS) {
        fprintf(stderr, "[sp_hex_sqfree] sp_hex_open failed 0x%x\n", rc);
        free(sess);
        sp_hex_sqfree_cache_free(sc);
        return -1;
    }
    sess->opened = true;
    sc->ctx      = (sp_hexagon_ctx_t *)sess;   // borrow the typed slot
    sc->owns_ctx = true;
    return 0;
}

void sp_hex_sqfree_cache_free(sp_hex_sqfree_cache_t *sc) {
    if (!sc) return;
    sp_hex_sqfree_session_t *sess = sess_from_cache(sc);
    if (sess) {
        if (sess->opened && sess->handle != (remote_handle64)-1) {
            sp_hex_close(sess->handle);
        }
        free(sess);
        sc->ctx = NULL;
    }
    if (sc->k_cache) {
        for (int s = 0; s < sc->n_slots; ++s) free(sc->k_cache[s]);
        free(sc->k_cache);
    }
    if (sc->v_cache) {
        for (int s = 0; s < sc->n_slots; ++s) free(sc->v_cache[s]);
        free(sc->v_cache);
    }
    free(sc->pad_scratch);
    free(sc->coeff_scratch);
    if (sc->residual_pad)  rpcmem_free(sc->residual_pad);
    if (sc->predicted_pad) rpcmem_free(sc->predicted_pad);
    if (sc->recon_pad)     rpcmem_free(sc->recon_pad);
    free(sc->calib_sum);
    free(sc->calib_sum2);
    free(sc->calib_cov);
    sp_knight_mask_free(&sc->mask);
    memset(sc, 0, sizeof(*sc));
}

// ============================================================================
// Calibration — variance-ranked Knight mask (mirrors sp_sqfree_calibrate_*).
// ============================================================================

int sp_hex_sqfree_calibrate_begin(sp_hex_sqfree_cache_t *sc) {
    if (!sc || sc->calibrating) return -1;
    const int pd = sc->pad_dim;
    sc->calib_sum  = (double *)calloc((size_t)pd, sizeof(double));
    sc->calib_sum2 = (double *)calloc((size_t)pd, sizeof(double));
    if (!sc->calib_sum || !sc->calib_sum2) {
        free(sc->calib_sum);  sc->calib_sum  = NULL;
        free(sc->calib_sum2); sc->calib_sum2 = NULL;
        return -1;
    }
    sc->calib_n     = 0;
    sc->calibrating = true;
    return 0;
}

void sp_hex_sqfree_calibrate_feed(sp_hex_sqfree_cache_t *sc, const float *vec) {
    if (!sc || !sc->calibrating || !vec) return;
    const int hd = sc->config.head_dim;
    const int pd = sc->pad_dim;
    sp_sqfree_pad_f32(vec, hd, sc->pad_scratch, pd);
    sp_vht2_forward_f32(sc->pad_scratch, pd);
    for (int i = 0; i < pd; ++i) {
        double v = (double)sc->pad_scratch[i];
        sc->calib_sum [i] += v;
        sc->calib_sum2[i] += v * v;
    }
    sc->calib_n++;
}

int sp_hex_sqfree_calibrate_end(sp_hex_sqfree_cache_t *sc) {
    if (!sc || !sc->calibrating || sc->calib_n < 1) return -1;
    sc->calibrating = false;
    const int pd = sc->pad_dim;
    /* Variance per coefficient. */
    float *var = (float *)calloc((size_t)pd, sizeof(float));
    if (!var) return -1;
    const double inv_n = 1.0 / (double)sc->calib_n;
    for (int i = 0; i < pd; ++i) {
        double mean = sc->calib_sum[i] * inv_n;
        double v    = sc->calib_sum2[i] * inv_n - mean * mean;
        if (v < 0.0) v = 0.0;
        var[i] = (float)v;
    }
    /* Rebuild knight mask with variance-ranked skeleton. */
    sp_knight_mask_free(&sc->mask);
    int rc = sp_knight_mask_init(&sc->mask, pd, SP_HEX_HIER_SKEL, var);
    free(var);
    free(sc->calib_sum);  sc->calib_sum  = NULL;
    free(sc->calib_sum2); sc->calib_sum2 = NULL;
    if (rc != 0) return -1;
    if (sc->mask.sk_k != SP_HEX_HIER_SKEL ||
        sc->mask.n_res != SP_HEX_HIER_LANES) {
        return -1;
    }
    return 0;
}

// ============================================================================
// Encode / decode helpers — single-position write/read.
// ============================================================================

static void sp_hex_sqfree_encode_one(sp_hex_sqfree_cache_t *sc,
                                      const float *vec,
                                      uint8_t *slot_ptr /* SLOT_BYTES */) {
    const int hd   = sc->config.head_dim;
    const int pd   = sc->pad_dim;
    const int n_sk = sc->mask.sk_k;       /* 14 */
    const int n_rs = sc->mask.n_res;      /* 140 */
    sp_hex_sqfree_session_t *sess = sess_from_cache(sc);
    if (!sess || sess->handle == (remote_handle64)-1) {
        memset(slot_ptr, 0, SP_HEX_SQFREE_SLOT_BYTES);
        return;
    }

    /* 1. Pad host -> pad_dim, VHT2 forward (host scalar). */
    sp_sqfree_pad_f32(vec, hd, sc->pad_scratch, pd);
    memcpy(sc->coeff_scratch, sc->pad_scratch, sizeof(float) * (size_t)pd);
    sp_vht2_forward_f32(sc->coeff_scratch, pd);

    /* 2. Extract skeleton (fp16 directly into slot bytes 0..27). */
    uint16_t *skel_fp16 = (uint16_t *)slot_ptr;
    float skel_fp32[SP_HEX_HIER_SKEL];
    for (int i = 0; i < n_sk; ++i) {
        float v = sc->coeff_scratch[sc->mask.skeleton_idx[i]];
        skel_fp32[i] = v;
        skel_fp16[i] = sp_f32_to_f16(v);
    }

    /* 3. Extract residual coefficients into the 160-padded buffer.  Tail
     *    20 lanes stay at zero so the DSP amax pass doesn't see garbage. */
    memset(sc->residual_pad, 0, SP_HEX_HIER_PAD * sizeof(float));
    for (int i = 0; i < n_rs; ++i) {
        sc->residual_pad[i] = sc->coeff_scratch[sc->mask.residual_idx[i]];
    }

    /* 4. FastRPC: predict.  predicted_pad first 140 lanes valid, tail 0. */
    memset(sc->predicted_pad, 0, SP_HEX_HIER_PAD * sizeof(float));
    int rc = sp_hex_hier_predict_f32(sess->handle,
                                       skel_fp32, n_sk,
                                       sc->predicted_pad, SP_HEX_HIER_LANES);
    if (rc != AEE_SUCCESS) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[sp_hex_sqfree] hier_predict rc=0x%x\n", rc);
            warned = 1;
        }
        memset(slot_ptr, 0, SP_HEX_SQFREE_SLOT_BYTES);
        return;
    }

    /* 5. FastRPC: residual quantize + spinor pack → packed[71] + amax. */
    uint8_t *packed = slot_ptr + SP_HEX_SQFREE_SKEL_BYTES;   /* offset 28 */
    float    amax   = 0.0f;
    rc = sp_hex_residual_quantize_spinor(sess->handle,
                                          sc->residual_pad, SP_HEX_HIER_PAD,
                                          sc->predicted_pad, SP_HEX_HIER_PAD,
                                          packed, SP_HEX_SQFREE_PACK_BYTES,
                                          &amax);
    if (rc != AEE_SUCCESS) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[sp_hex_sqfree] residual_quantize_spinor rc=0x%x\n", rc);
            warned = 1;
        }
        memset(slot_ptr, 0, SP_HEX_SQFREE_SLOT_BYTES);
        return;
    }
    /* Store amax at bytes 99..102. */
    memcpy(slot_ptr + SP_HEX_SQFREE_SKEL_BYTES + SP_HEX_SQFREE_PACK_BYTES,
           &amax, sizeof(float));
}

static void sp_hex_sqfree_decode_one(const sp_hex_sqfree_cache_t *sc,
                                      const uint8_t *slot_ptr,
                                      float *out_vec) {
    const int hd   = sc->config.head_dim;
    const int pd   = sc->pad_dim;
    const int n_sk = sc->mask.sk_k;
    const int n_rs = sc->mask.n_res;
    sp_hex_sqfree_session_t *sess = sess_from_cache(sc);
    if (!sess || sess->handle == (remote_handle64)-1) {
        memset(out_vec, 0, sizeof(float) * (size_t)hd);
        return;
    }

    /* 1. Recover skel_fp32 from slot bytes 0..27. */
    const uint16_t *skel_fp16 = (const uint16_t *)slot_ptr;
    float skel_fp32[SP_HEX_HIER_SKEL];
    for (int i = 0; i < n_sk; ++i) skel_fp32[i] = sp_f16_to_f32(skel_fp16[i]);

    /* 2. Pull amax and packed pointer. */
    const uint8_t *packed = slot_ptr + SP_HEX_SQFREE_SKEL_BYTES;
    float amax = 0.0f;
    memcpy(&amax, slot_ptr + SP_HEX_SQFREE_SKEL_BYTES + SP_HEX_SQFREE_PACK_BYTES,
           sizeof(float));

    /* 3. FastRPC: decode (predict + unpack + add). */
    sp_hex_sqfree_cache_t *sc_mut = (sp_hex_sqfree_cache_t *)sc;
    int rc = sp_hex_hier_decode_f32(sess->handle,
                                      skel_fp32, n_sk,
                                      packed, SP_HEX_SQFREE_PACK_BYTES,
                                      amax,
                                      sc_mut->recon_pad, SP_HEX_HIER_LANES);
    if (rc != AEE_SUCCESS) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[sp_hex_sqfree] hier_decode rc=0x%x\n", rc);
            warned = 1;
        }
        memset(out_vec, 0, sizeof(float) * (size_t)hd);
        return;
    }

    /* 4. Scatter skeleton + reconstructed residual into pad_dim layout. */
    float *pad = sc_mut->pad_scratch;
    memset(pad, 0, sizeof(float) * (size_t)pd);
    for (int i = 0; i < n_sk; ++i) pad[sc->mask.skeleton_idx[i]] = skel_fp32[i];
    for (int i = 0; i < n_rs; ++i) pad[sc->mask.residual_idx [i]] = sc_mut->recon_pad[i];

    /* 5. Inverse VHT2 (self-inverse) + unpad. */
    sp_vht2_forward_f32(pad, pd);
    sp_sqfree_unpad_f32(pad, out_vec, hd);
}

// ============================================================================
// Public write/read entry points — same signatures as sp_sqfree_*.
// ============================================================================

void sp_hex_sqfree_write_k(sp_hex_sqfree_cache_t *sc, int layer, int head,
                            int pos, const float *k_vec) {
    if (!sc || !k_vec || pos < 0 || pos >= sc->max_seq_len) return;
    const int slot = layer * sc->config.n_heads_kv + head;
    if (slot < 0 || slot >= sc->n_slots) return;
    uint8_t *p = sc->k_cache[slot] + (size_t)pos * SP_HEX_SQFREE_SLOT_BYTES;
    sp_hex_sqfree_encode_one(sc, k_vec, p);
}

void sp_hex_sqfree_write_v(sp_hex_sqfree_cache_t *sc, int layer, int head,
                            int pos, const float *v_vec) {
    if (!sc || !v_vec || pos < 0 || pos >= sc->max_seq_len) return;
    const int slot = layer * sc->config.n_heads_kv + head;
    if (slot < 0 || slot >= sc->n_slots) return;
    uint8_t *p = sc->v_cache[slot] + (size_t)pos * SP_HEX_SQFREE_SLOT_BYTES;
    sp_hex_sqfree_encode_one(sc, v_vec, p);
}

void sp_hex_sqfree_read_k(const sp_hex_sqfree_cache_t *sc, int layer, int head,
                           int pos, float *k_out) {
    if (!sc || !k_out || pos < 0 || pos >= sc->max_seq_len) {
        if (k_out && sc) memset(k_out, 0, sizeof(float) * sc->config.head_dim);
        return;
    }
    const int slot = layer * sc->config.n_heads_kv + head;
    if (slot < 0 || slot >= sc->n_slots) {
        memset(k_out, 0, sizeof(float) * sc->config.head_dim);
        return;
    }
    const uint8_t *p = sc->k_cache[slot] + (size_t)pos * SP_HEX_SQFREE_SLOT_BYTES;
    sp_hex_sqfree_decode_one(sc, p, k_out);
}

void sp_hex_sqfree_read_v(const sp_hex_sqfree_cache_t *sc, int layer, int head,
                           int pos, float *v_out) {
    if (!sc || !v_out || pos < 0 || pos >= sc->max_seq_len) {
        if (v_out && sc) memset(v_out, 0, sizeof(float) * sc->config.head_dim);
        return;
    }
    const int slot = layer * sc->config.n_heads_kv + head;
    if (slot < 0 || slot >= sc->n_slots) {
        memset(v_out, 0, sizeof(float) * sc->config.head_dim);
        return;
    }
    const uint8_t *p = sc->v_cache[slot] + (size_t)pos * SP_HEX_SQFREE_SLOT_BYTES;
    sp_hex_sqfree_decode_one(sc, p, v_out);
}

#endif // SP_HEXAGON_FASTRPC
