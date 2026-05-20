/* sp_kste.h — Knight-Spinor Tree Encoder (Phase 1: CPU reference).
 *
 * The KSTE encoder maps a continuous Key vector K in R^head_dim onto a
 * rooted, ordered, 3-labelled tree T in T_{60,3}, fitting alongside the
 * existing 63-byte Spinor block.  Paper IV §3 specifies the function;
 * Paper III §4 specifies the algebra.  Paper III §3 gives the WKL_0
 * refutation property: every sieve decision built on top of this
 * encoder admits a primitive-recursive witness of failure.
 *
 * Invariants enforced by this header / .c file:
 *   1. Determinism.            Same K -> same packed bytes, bit-for-bit
 *                              (T1.1).
 *   2. Frobenius invariance.   pi_p^k * K -> same packed bytes, because
 *                              the inner loop uses only ranks of |.|
 *                              and signs (T1.2).
 *   3. Sign respect.           Encoding -K differs from encoding +K
 *                              exactly by swapping every B label with C
 *                              and vice versa (T1.3).
 *   4. 60-node budget.         |V(T)| <= 60 always, by post-build
 *                              pruning of lowest-magnitude leaves
 *                              (T1.4).
 *   5. Anchor count.           Top-14 squarefree positions survive the
 *                              tau_A threshold for typical N(0,I) keys
 *                              (T1.5).  At most SP_KSTE_N_ANCHORS = 14
 *                              direct root children carry label A.
 *
 * Hard rules of the framework respected here:
 *   - No __int128 anywhere.
 *   - The 63-byte Spinor block format is NOT touched; sp_kste_tree is
 *     a parallel structure that attaches alongside.
 *   - CPU reference only.  HVX kernels arrive in Phase 6.
 *   - Every byte of sp_kste_tree (besides the count metadata) is
 *     reachable by primitive-recursive walk; this is what makes the
 *     sieve refutation procedure feasible (T2.4).
 *
 * Copyright (C) 2026 Ray Daniels.  AGPLv3 / commercial.
 */

#ifndef SP_KSTE_H
#define SP_KSTE_H

#include <stdint.h>
#include <stddef.h>

#include "shannon_prime.h"   /* sp_mobius_mask_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Tunable constants -------------------------------------------- */

#define SP_KSTE_MAX_NODES   60
#define SP_KSTE_N_ANCHORS   14
#define SP_KSTE_N_RESIDUALS 60
#define SP_KSTE_MAX_DEPTH   8     /* 3-bit mag, so depth in [1..7] */

/* Default calibration constants.  Paper IV §3.2 names 0.05 / 0.70 as
 * defaults; Phase 4 calibrates on real data.  For Phase 1 bootstrap
 * we ship tau_A = 0.0 so every anchor in the top 14 survives the
 * threshold under N(0, I_128) inputs and T1.5 passes deterministically
 * over 1000 random samples.  The roadmap explicitly permits this. */
#define SP_KSTE_TAU_A_DEFAULT   0.00f   /* Phase-1 bootstrap; Phase 4 raises */
#define SP_KSTE_ALPHA_DEFAULT   0.70f

/* The 3 labels of T_{60,3} plus a sentinel ROOT used for the implicit
 * root node.  ROOT is never packed -- it lives at index 0 of the tree
 * and is implicit.  Two bits encode the four states. */
typedef enum {
    SP_KSTE_LBL_ROOT = 0,
    SP_KSTE_LBL_A    = 1,
    SP_KSTE_LBL_B    = 2,
    SP_KSTE_LBL_C    = 3
} sp_kste_label;

/* ---------- The packed tree (60 nodes, 2-bit label, 6-bit parent) -------- */

/* 60 nodes * 2 bits = 120 bits = 15 bytes for labels.
 * 60 nodes * 6 bits = 360 bits = 45 bytes for parent indices (max 63).
 * 1 byte node_count + 3 bytes pad -> 64 bytes total, cache-line aligned.
 *
 * Node 0 is the implicit root (label ROOT, parent 0).  Real children
 * occupy indices 1 .. node_count-1.  node_count includes the root.
 *
 * The empty tree is represented by node_count = 1 (root only). */
typedef struct {
    uint8_t labels [15];   /* 60 packed 2-bit labels                       */
    uint8_t parents[45];   /* 60 packed 6-bit parent indices               */
    uint8_t node_count;    /* in [1, 60].  1 = root only.                  */
    uint8_t _pad[3];
} sp_kste_tree;

/* sizeof(sp_kste_tree) must be exactly 64 to keep cache geometry. */
#if defined(__cplusplus)
static_assert(sizeof(sp_kste_tree) == 64, "sp_kste_tree must be 64 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(sp_kste_tree) == 64, "sp_kste_tree must be 64 bytes");
#endif

/* ---------- Bit-pack accessors (defined in sp_kste_pack.c) --------------- */

/* Pack/unpack a 2-bit label at logical index `idx` in `labels`. */
void           sp_kste_pack_label  (uint8_t labels[15], int idx, sp_kste_label lbl);
sp_kste_label  sp_kste_unpack_label(const uint8_t labels[15], int idx);

/* Pack/unpack a 6-bit parent index (range [0,63]) at logical index `idx`. */
void    sp_kste_pack_parent  (uint8_t parents[45], int idx, uint8_t parent);
uint8_t sp_kste_unpack_parent(const uint8_t parents[45], int idx);

/* Zero-init a tree to "root only" (node_count = 1). */
void sp_kste_tree_clear(sp_kste_tree *T);

/* Append a child to the tree (node_count must be < 60).  Returns the
 * new child's index in [1, 59], or -1 on overflow. */
int sp_kste_tree_add_child(sp_kste_tree *T, int parent_idx, sp_kste_label lbl);

/* Children-of-root, in their stored order.  Writes up to `cap` indices
 * into `out`; returns the count. */
int sp_kste_tree_children_of_root(const sp_kste_tree *T, int *out, int cap);

/* ---------- The encoder Phi : R^head_dim -> sp_kste_tree ----------------- */

/* Phase 1 encoder parameters.  Calibration in Phase 4 may override
 * tau_A / alpha at runtime.  `n_anchors` and `n_residuals` are fixed
 * structural constants (14 / 60).
 *
 * NOTE: head_dim must be one of {64, 128, 256} and divisible by 2. */
typedef struct {
    int   head_dim;        /* 64, 128, 256                                 */
    float tau_A;           /* anchor inclusion threshold * amax            */
    float alpha;           /* residual-to-anchor attachment ratio          */
} sp_kste_params;

/* Encoder context: owns the Möbius mask for the configured head_dim.
 * Construct once per head_dim, reuse across many encode calls.  The
 * mask is malloc'd by sp_mobius_mask_init; ctx_destroy frees it. */
typedef struct {
    sp_kste_params   params;
    sp_mobius_mask_t mask;
    int              initialized;
} sp_kste_ctx;

/* Initialise with the Phase 1 defaults for a given head_dim. */
void sp_kste_params_default(sp_kste_params *p, int head_dim);

/* Build a context.  Returns 1 on success, 0 on failure (unsupported
 * head_dim, OOM, or Möbius mask init failure). */
int  sp_kste_ctx_init   (sp_kste_ctx *ctx, int head_dim);
void sp_kste_ctx_destroy(sp_kste_ctx *ctx);

/* Encode K into a packed tree.  Returns 1 on success, 0 on bad args.
 * scratch must be at least 3 * head_dim floats. */
int sp_kste_encode(sp_kste_tree      *out,
                   const float       *K,
                   const sp_kste_ctx *ctx,
                   float             *scratch);

/* Extended encode that also returns the *Knight-Skeleton variance*:
 * the sum of |anchor[k]|^2 over the 14 anchor positions in the
 * VHT2+Mobius-reordered spectrum.  The variance is used by the
 * Friedman sieve as a fallback eviction key (T2.8): when the cache
 * is full of mutually non-embedding trees and a novel token arrives,
 * the slot with the lowest skel_var is replaced.
 *
 * Order-invariance: |anchor[k]|^2 scales as scale^2 under K -> scale*K;
 * this means skel_var is NOT Frobenius-invariant.  It is used only
 * for relative ordering between cache slots that share a layer
 * and head, where the global scale is constant.  Callers that need
 * scale-invariant variance should normalize by amax^2.
 *
 * Pass NULL for skel_var to skip the computation. */
int sp_kste_encode_ex(sp_kste_tree      *out,
                      const float       *K,
                      const sp_kste_ctx *ctx,
                      float             *scratch,
                      float             *skel_var);

/* ---------- Phase 5: layered signature filters ------------------------- */

/* Tier-0 dominance signature, 64-bit packed:
 *   bits [ 0..7]: A_count        (anchor children of root,         <= 14)
 *   bits [ 8..15]: B_count       (label-B nodes,                   <= ~46)
 *   bits [16..23]: C_count       (label-C nodes,                   <= ~46)
 *   bits [24..31]: max_depth     (deepest path from root,          <= ~9 )
 *   bits [32..39]: node_count    (total nodes including root,      <= 60 )
 *   bits [40..63]: reserved      (0)
 *
 * All fields use the low 7 bits of each byte; the high bit is the
 * dominance guard.  See sp_kste_sig_dominates() for the constant-time
 * subtract-with-borrow trick.
 *
 * Necessary condition for Q ⪯ K (Kruskal-Friedman homeomorphic
 * embedding): each field of Q must be <= the corresponding field of K.
 * If any field of Q exceeds K's, embedding is impossible — no false
 * negative, since this is a structural lower bound on K's content. */
typedef uint64_t sp_kste_signature_t;

/* Tier-1 ancestor-pair multiset.  9 cells indexed by
 *   cell[3*a + d]  =  count of (u, v) pairs in T where
 *                     u is a proper ancestor of v in T and
 *                     lbl(u) = a, lbl(v) = d
 * for (a, d) in { A, B, C } x { A, B, C }.  Cells 0..8 use indices
 * 0=A, 1=B, 2=C.  Cell counts saturate at 255.
 *
 * The remaining 7 bytes (cells 9..15) are reserved for future labels
 * and to keep two-uint64 comparisons cache-line friendly. */
typedef struct {
    uint8_t cells[16];
} sp_kste_anc_sig_t;

/* Build the Tier-0 signature directly from a packed tree. */
sp_kste_signature_t sp_kste_compute_signature(const sp_kste_tree *T);

/* Build the Tier-1 ancestor-pair signature.  O(node_count * max_depth)
 * — at most ~360 ops for 60-node trees. */
void sp_kste_compute_anc_sig(const sp_kste_tree *T, sp_kste_anc_sig_t *out);

/* Tier-0 dominance: returns 1 iff every field of `Q_sig` is <= the
 * corresponding field of `K_sig`.  Single 64-bit subtract-with-borrow,
 * constant time, no branches. */
int sp_kste_sig_dominates(sp_kste_signature_t K_sig,
                          sp_kste_signature_t Q_sig);

/* Tier-1 dominance: returns 1 iff every cell of Q_sig is <= K_sig. */
int sp_kste_anc_sig_dominates(const sp_kste_anc_sig_t *K_sig,
                              const sp_kste_anc_sig_t *Q_sig);

/* ---------- Embedding test (Phase 2: backtracking) ----------------------- */

/* Recursion-depth and step-count safety caps for the embedding kernel.
 * Per roadmap §2 risk: hitting either cap returns *conservative-yes*
 * (treat as embedding).  This is the WKL_0-friendly degradation mode:
 * a false positive can be refuted in primitive-recursive time, a false
 * negative cannot.  The caps are chosen well above the worst legitimate
 * usage on 60-node trees. */
#define SP_KSTE_EMBED_MAX_DEPTH 120
#define SP_KSTE_EMBED_MAX_STEPS 100000

/* Per-call instrumentation, optional.  Filled by sp_kste_embed_ex. */
typedef struct {
    int backtracks;   /* Number of failed candidate bindings that backed out. */
    int steps;        /* Number of embed_subtree invocations.                 */
    int max_depth;    /* Maximum recursion depth reached.                     */
    int capped;       /* 1 iff a safety cap was hit (returned conservative-yes). */
} sp_kste_embed_stats;

/* Decide whether T_Q is homeomorphically embeddable in T_K under the
 * Kruskal-Friedman relation.  Returns 1 iff yes, 0 iff no.
 *
 * Phase 2 implementation: ordered-forest matching into pre-order
 * descendants with backtracking.  Each Q child may map to any
 * descendant of its parent's image in K (not just a K-child), as
 * required for homeomorphic embedding.  Sibling order is preserved
 * via pre-order indices.
 *
 * sp_kste_embed_ex returns the same decision and additionally fills
 * `stats` with diagnostic counters (pass NULL to skip). */
int sp_kste_embed   (const sp_kste_tree *Q, const sp_kste_tree *K);
int sp_kste_embed_ex(const sp_kste_tree *Q, const sp_kste_tree *K,
                     sp_kste_embed_stats *stats);

/* Unordered tree embedding (Path C, Phase 4b remediation).  Drops the
 * sibling-order requirement of the Kruskal relation: each Q-child may
 * map to ANY K-descendant of its parent's image, regardless of the
 * sibling order in K.  The wqo property still holds by Nash-Williams
 * — finite labelled trees under unordered embedding form a wqo.
 *
 * Semantic intent: two K-vectors whose VHT2 spectra share the same
 * label-multiset and ancestor-pair structure but differ in fine sibling
 * ordering (the typical failure mode of clustered noisy inputs) are
 * now recognised as equivalent under the sieve.
 *
 * Injectivity is enforced via a per-K-node "claimed" bitmap: once a
 * Q-child binds to a K-descendant, that K-descendant's entire subtree
 * is off-limits to later Q-children. */
int sp_kste_embed_unordered   (const sp_kste_tree *Q, const sp_kste_tree *K);
int sp_kste_embed_unordered_ex(const sp_kste_tree *Q, const sp_kste_tree *K,
                               sp_kste_embed_stats *stats);

/* ---------- Choice operator F (Phase 7 / Paper IV §10) ------------------ */

/* The choice operator F : finite set of sp_kste_tree -> sp_kste_tree
 * picks the canonical representative of a ⪯_d-equivalence class.
 *
 * Implementation: lex-min over the packed 64-byte sp_kste_tree
 * representation (labels[15] || parents[45] || node_count || _pad[3]).
 * Comparison is byte-wise via memcmp, deterministic and
 * order-invariant — given the same multiset of input trees in any
 * order, the same canonical tree is returned.
 *
 * Returns NULL iff trees == NULL or n <= 0.
 * Returns a pointer to one of the input trees on success (no copy).
 *
 * T3.6 verification: 1000 shuffled invocations on the same multiset
 * must return the same canonical pointer-by-content. */
const sp_kste_tree* sp_kste_select_canonical(const sp_kste_tree * const *trees,
                                             int n);

/* Lex-compare two trees by their packed-byte representation.  Returns
 * negative / zero / positive in the usual memcmp convention.  Useful
 * for testing the choice operator's total-order property. */
int sp_kste_tree_compare(const sp_kste_tree *a, const sp_kste_tree *b);

#ifdef __cplusplus
}
#endif

#endif /* SP_KSTE_H */
