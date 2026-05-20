/* sp_kste.c — Knight-Spinor Tree Encoder (Phase 1: CPU reference).
 *
 * See sp_kste.h for the invariants and Paper IV §3 / Paper III §4 for
 * the math.  Build order in this file:
 *
 *   sp_kste_params_default   — bootstrap calibration
 *   sp_kste_ctx_init/destroy — owns the Möbius mask
 *   sp_kste_encode           — VHT2 → Möbius → anchors → residuals → budget
 *   sp_kste_embed            — Phase-1 greedy ordered-tree embedding
 *
 * The encoder uses *only* ranks and signs in the inner loop after the
 * VHT2+Möbius transform.  Combined with the linearity of VHT2, this
 * gives the encoder its Frobenius invariance: any positive rescale of
 * K (in particular |pi_p^k|) preserves the rank pattern of |Y'| and
 * the sign pattern of Y'.  This is the load-bearing property for
 * T1.2 and for the WKL_0 refutation framework downstream.
 *
 * No __int128.  No SIMD.  No malloc inside encode (mask is in ctx).
 */

#include "sp_kste.h"
#include "shannon_prime.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Defaults / ctor ----------------------------------------------- */

void sp_kste_params_default(sp_kste_params *p, int head_dim)
{
    if (!p) return;
    p->head_dim = head_dim;
    p->tau_A    = SP_KSTE_TAU_A_DEFAULT;
    p->alpha    = SP_KSTE_ALPHA_DEFAULT;
}

int sp_kste_ctx_init(sp_kste_ctx *ctx, int head_dim)
{
    if (!ctx) return 0;
    /* Supported: 64, 128, 256.  VHT2 requires factor in {2,3,5,7,11}. */
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) return 0;
    memset(ctx, 0, sizeof(*ctx));
    sp_kste_params_default(&ctx->params, head_dim);
    /* Headroom: 14 anchors + 60 residuals = 74 indices required. */
    if (head_dim < 74) return 0;
    if (sp_mobius_mask_init(&ctx->mask, head_dim) != 0) {
        sp_mobius_mask_free(&ctx->mask);
        return 0;
    }
    ctx->initialized = 1;
    return 1;
}

void sp_kste_ctx_destroy(sp_kste_ctx *ctx)
{
    if (!ctx || !ctx->initialized) return;
    sp_mobius_mask_free(&ctx->mask);
    ctx->initialized = 0;
}

/* ---------- Sort helpers (small N, deterministic) ------------------------ */

/* Descending sort of indices by abs(values[idx]); insertion sort is
 * trivially deterministic and fast for our N <= 60.  Stable on ties: a
 * tie is broken by *lower index wins*, which is again deterministic
 * across platforms. */
static void sp_kste_sort_idx_by_abs_desc(int *idx, int n, const float *values)
{
    for (int i = 1; i < n; ++i) {
        int   ii  = idx[i];
        float iv  = fabsf(values[ii]);
        int   j   = i - 1;
        while (j >= 0) {
            float jv = fabsf(values[idx[j]]);
            /* Strict greater for descending; on tie keep lower idx first */
            if (jv > iv) break;
            if (jv == iv && idx[j] < ii) break;
            idx[j + 1] = idx[j];
            --j;
        }
        idx[j + 1] = ii;
    }
}

/* ---------- The encoder -------------------------------------------------- */

int sp_kste_encode_ex(sp_kste_tree      *out,
                      const float       *K,
                      const sp_kste_ctx *ctx,
                      float             *scratch,
                      float             *skel_var)
{
    if (skel_var) *skel_var = 0.0f;
    if (!out || !K || !ctx || !scratch || !ctx->initialized) return 0;

    const int hd = ctx->params.head_dim;
    if (hd != 64 && hd != 128 && hd != 256) return 0;

    /* scratch layout:
     *   [0 .. hd)        — VHT2 working buffer (will hold Y' after reorder)
     *   [hd .. 2*hd)     — Möbius reorder scratch
     *   [2*hd .. 3*hd)   — unused (reserved for Phase 2/3)
     */
    float *Y       = scratch;
    float *mscr    = scratch + hd;

    /* 1. Copy K into Y, then VHT2 in place. */
    memcpy(Y, K, sizeof(float) * (size_t)hd);
    sp_vht2_forward_f32(Y, hd);

    /* 2. Möbius reorder: squarefree indices to the front. */
    sp_mobius_reorder_ex(Y, &ctx->mask, mscr);

    /* 3. Compute global amax over Y[0 .. hd) for residual quantization. */
    float amax = 0.0f;
    for (int i = 0; i < hd; ++i) {
        float a = fabsf(Y[i]);
        if (a > amax) amax = a;
    }
    /* Guard against degenerate input (all zeros). */
    if (!(amax > 0.0f)) {
        sp_kste_tree_clear(out);
        return 1;     /* trivial tree: root only */
    }

    /* 4. Build the tree.  Root is node 0, implicit, label ROOT. */
    sp_kste_tree_clear(out);

    /* 4a. Anchors: positions 0 .. SP_KSTE_N_ANCHORS-1 in Möbius order.
     *
     * Per Paper IV §3.1 we order them by descending |anchor| (i.e., by
     * rank), and admit each one only if it exceeds tau_A * amax.  With
     * Phase-1 default tau_A = 0.0 every nonzero anchor is admitted. */
    int   anchor_idx[SP_KSTE_N_ANCHORS];
    int   n_anchor_input = SP_KSTE_N_ANCHORS;
    for (int k = 0; k < n_anchor_input; ++k) anchor_idx[k] = k;
    sp_kste_sort_idx_by_abs_desc(anchor_idx, n_anchor_input, Y);

    /* Knight-Skeleton variance: sum |anchor[k]|^2 over the 14 anchor
     * positions.  Computed here so it works whether or not the anchor
     * eventually clears the tau_A threshold. */
    if (skel_var) {
        float v = 0.0f;
        for (int k = 0; k < n_anchor_input; ++k) {
            float a = Y[anchor_idx[k]];
            v += a * a;
        }
        *skel_var = v;
    }

    int   anchor_root_child[SP_KSTE_N_ANCHORS];  /* tree node indices */
    int   n_anchors_actual = 0;
    const float anchor_thresh = ctx->params.tau_A * amax;

    for (int k = 0; k < n_anchor_input; ++k) {
        int src_pos = anchor_idx[k];
        if (!(fabsf(Y[src_pos]) > anchor_thresh)) break;
        int new_idx = sp_kste_tree_add_child(out, /*parent=*/0,
                                             SP_KSTE_LBL_A);
        if (new_idx < 0) break;
        anchor_root_child[n_anchors_actual++] = new_idx;
    }

    /* 4b. Residuals: positions SP_KSTE_N_ANCHORS .. (N_ANCHORS+N_RESIDUALS).
     *
     * Quantize each |Y'[j]| to 3 bits relative to amax, label by sign.
     * Process in descending magnitude order so the strongest residuals
     * claim budget first. */
    int   res_src[SP_KSTE_N_RESIDUALS];  /* indices into Y (offset by 14) */
    for (int j = 0; j < SP_KSTE_N_RESIDUALS; ++j) {
        res_src[j] = SP_KSTE_N_ANCHORS + j;
    }
    /* Sort by descending |Y[idx]|.  Stable on ties (lower idx wins). */
    sp_kste_sort_idx_by_abs_desc(res_src, SP_KSTE_N_RESIDUALS, Y);

    /* For attach_idx we use rank within the sorted residual sequence.
     * Stronger residuals attach to higher-ranked anchors; the formula
     * is order-invariant.  rank_j ranges over 0 .. 59 in sorted order. */
    const float alpha = ctx->params.alpha;

    for (int rank_j = 0; rank_j < SP_KSTE_N_RESIDUALS; ++rank_j) {
        if (out->node_count >= SP_KSTE_MAX_NODES) break;

        int   src   = res_src[rank_j];
        float v     = Y[src];
        float av    = fabsf(v);
        /* 3-bit quantisation: nearest level in 0..7 (0 = below noise floor). */
        int   mag   = (int)floorf((av / amax) * 7.0f + 0.5f);
        if (mag <= 0) continue;
        if (mag > 7) mag = 7;

        sp_kste_label lbl = (v >= 0.0f) ? SP_KSTE_LBL_B : SP_KSTE_LBL_C;

        /* Choose the dominating anchor.  Path-B bucketed attachment
         * (Phase 4b remediation): the 60 residuals fall into 4 rank
         * buckets of 15 each; each bucket maps to a single anchor.
         * Small rank perturbations between similar K-vectors stay
         * inside the same bucket and therefore attach to the SAME
         * anchor — the topology no longer shatters at rank boundaries.
         * alpha now controls the anchor-span: alpha=1.0 spreads buckets
         * across the full anchor range; alpha=0.3 collapses them onto
         * the strongest few anchors. */
        int attach_idx = 0;
        if (n_anchors_actual > 0) {
            int rank_bucket = rank_j / 15;            /* 0..3 for 0..59 */
            if (rank_bucket > 3) rank_bucket = 3;
            int anchor_span = (int)((double)alpha * (double)n_anchors_actual);
            if (anchor_span < 1) anchor_span = 1;
            attach_idx = (rank_bucket * anchor_span) / 4;
            if (attach_idx >= n_anchors_actual) attach_idx = n_anchors_actual - 1;
        }
        int parent = (n_anchors_actual > 0)
                   ? anchor_root_child[attach_idx]
                   : /*root=*/0;

        /* 4c. Magnitude-becomes-depth: chain of `mag` nodes from parent. */
        for (int d = 0; d < mag; ++d) {
            if (out->node_count >= SP_KSTE_MAX_NODES) break;
            int new_idx = sp_kste_tree_add_child(out, parent, lbl);
            if (new_idx < 0) break;
            parent = new_idx;
        }
    }

    /* The budget is enforced at construction time by the
     * `out->node_count >= MAX_NODES` checks above.  This matches the
     * Paper IV §3.1 spec at the limit (strongest residuals claim
     * budget first), and avoids the post-build prune step which is
     * harder to make order-invariant. */
    return 1;
}


/* Public wrapper: encode without returning the variance. */
int sp_kste_encode(sp_kste_tree      *out,
                   const float       *K,
                   const sp_kste_ctx *ctx,
                   float             *scratch)
{
    return sp_kste_encode_ex(out, K, ctx, scratch, NULL);
}

/* The embedding kernel lives in sp_kste_embed.c (Phase 2 backtracking
 * over pre-order descendants).  This file owns only the encoder. */

/* ---------- Phase 5: Tier-0 / Tier-1 signature filters ------------------ */

sp_kste_signature_t sp_kste_compute_signature(const sp_kste_tree *T)
{
    if (!T) return 0;
    int a = 0, b = 0, c = 0;
    int n = T->node_count;
    int depth[SP_KSTE_MAX_NODES];
    depth[0] = 0;
    int max_d = 0;
    for (int i = 1; i < n; ++i) {
        int p = sp_kste_unpack_parent(T->parents, i);
        if (p < 0 || p >= i) { depth[i] = 0; }
        else                  depth[i] = depth[p] + 1;
        if (depth[i] > max_d) max_d = depth[i];

        sp_kste_label lbl = sp_kste_unpack_label(T->labels, i);
        if      (lbl == SP_KSTE_LBL_A) ++a;
        else if (lbl == SP_KSTE_LBL_B) ++b;
        else if (lbl == SP_KSTE_LBL_C) ++c;
    }
    /* Clamp each field to 7 bits so the dominance trick stays valid. */
    if (a     > 127) a     = 127;
    if (b     > 127) b     = 127;
    if (c     > 127) c     = 127;
    if (max_d > 127) max_d = 127;
    if (n     > 127) n     = 127;

    sp_kste_signature_t sig = 0;
    sig |= ((uint64_t)(uint8_t)a)     <<  0;
    sig |= ((uint64_t)(uint8_t)b)     <<  8;
    sig |= ((uint64_t)(uint8_t)c)     << 16;
    sig |= ((uint64_t)(uint8_t)max_d) << 24;
    sig |= ((uint64_t)(uint8_t)n)     << 32;
    return sig;
}

void sp_kste_compute_anc_sig(const sp_kste_tree *T, sp_kste_anc_sig_t *out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!T) return;

    int n = T->node_count;
    /* For each node v with a non-root label, walk up the parent chain
     * and bump cell[3*a + d] for each proper ancestor u (also non-root). */
    for (int v = 1; v < n; ++v) {
        sp_kste_label v_lbl = sp_kste_unpack_label(T->labels, v);
        if (v_lbl < SP_KSTE_LBL_A) continue;
        int d_idx = (int)v_lbl - 1;       /* A=0, B=1, C=2 */

        int u = (int)sp_kste_unpack_parent(T->parents, v);
        while (u > 0) {
            sp_kste_label u_lbl = sp_kste_unpack_label(T->labels, u);
            if (u_lbl >= SP_KSTE_LBL_A) {
                int a_idx = (int)u_lbl - 1;
                int cell  = 3 * a_idx + d_idx;
                if (out->cells[cell] < 255) out->cells[cell]++;
            }
            u = (int)sp_kste_unpack_parent(T->parents, u);
        }
    }
}

int sp_kste_sig_dominates(sp_kste_signature_t K_sig, sp_kste_signature_t Q_sig)
{
    /* Single-instruction dominance:
     *   set high bit of every K byte, clear high bit of every Q byte,
     *   subtract.  If any Q-byte exceeded its K-byte, that byte underflows
     *   and its high bit clears.  All-high means all-dominated. */
    const uint64_t HI = 0x8080808080808080ULL;
    const uint64_t LO = 0x7F7F7F7F7F7F7F7FULL;
    uint64_t diff = (K_sig | HI) - (Q_sig & LO);
    return (diff & HI) == HI;
}

int sp_kste_anc_sig_dominates(const sp_kste_anc_sig_t *K_sig,
                              const sp_kste_anc_sig_t *Q_sig)
{
    if (!K_sig || !Q_sig) return 0;
    uint64_t K0, K1, Q0, Q1;
    memcpy(&K0, &K_sig->cells[0], 8);
    memcpy(&K1, &K_sig->cells[8], 8);
    memcpy(&Q0, &Q_sig->cells[0], 8);
    memcpy(&Q1, &Q_sig->cells[8], 8);
    return sp_kste_sig_dominates(K0, Q0) && sp_kste_sig_dominates(K1, Q1);
}
