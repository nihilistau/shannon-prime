/* sp_kste_embed.c — Phase 2 homeomorphic-embedding kernel.
 *
 * Decides T_Q ⪯ T_K under the Kruskal-Friedman relation (Paper III §2.2):
 *
 *   1. Label preservation       lambda_K(iota(v)) = lambda_Q(v).
 *   2. Ancestor preservation    u ancestor of v in Q  =>
 *                               iota(u) ancestor of iota(v) in K.
 *   3. Order preservation       sibling order preserved via pre-order
 *                               indices in K.
 *
 * Algorithm: per-tree views (children lists, pre-order, subtree sizes)
 * are computed once; the decision recurses through embed_subtree and
 * match_children with full backtracking.  Each Q-child may map to any
 * descendant of its parent's image in K, not just a K-child.
 *
 * Safety: hard caps on recursion depth (SP_KSTE_EMBED_MAX_DEPTH) and
 * step count (SP_KSTE_EMBED_MAX_STEPS).  Hitting either returns
 * conservative-yes — a false positive that the WKL_0 refutation
 * procedure can locate finitely.  Per roadmap §2 risk.
 *
 * No allocations on the hot path; views live on the stack of
 * sp_kste_embed_ex.  Phase 6's HVX kernel replaces the inner loop;
 * this file is the CPU reference required by the cross-phase
 * "CPU & HVX agree bit-exactly" invariant.
 */

#include "sp_kste.h"

#include <string.h>

/* ---------- Per-tree view ------------------------------------------------ */

typedef struct {
    /* Children stored compactly in build (= parent-array) order. */
    int     child_start[SP_KSTE_MAX_NODES + 1];   /* CSR-style offsets        */
    int     n_children [SP_KSTE_MAX_NODES];
    int     child_idx  [SP_KSTE_MAX_NODES];       /* flattened child list     */

    /* Pre-order DFS over the children-in-stored-order. */
    int     pre        [SP_KSTE_MAX_NODES];       /* pre[node] = pre-order ix */
    int     size       [SP_KSTE_MAX_NODES];       /* subtree size             */
    int     pre_inv    [SP_KSTE_MAX_NODES];       /* pre_inv[i] = node at i   */
    int     node_count;
} sp_kste_view;

/* Build child-list (CSR) from the packed parent array.  Children of node p
 * appear in the order their indices appear in [1, node_count) — that is
 * the same order in which the encoder inserted them.  This is the
 * canonical sibling order for the order-preservation rule above. */
static void sp_kste_build_children(const sp_kste_tree *T, sp_kste_view *V)
{
    int nc = T->node_count;
    if (nc < 0) nc = 0;
    if (nc > SP_KSTE_MAX_NODES) nc = SP_KSTE_MAX_NODES;
    V->node_count = nc;

    /* Count children per parent. */
    memset(V->n_children, 0, sizeof(V->n_children));
    for (int i = 1; i < nc; ++i) {
        int p = sp_kste_unpack_parent(T->parents, i);
        if (p >= 0 && p < nc) V->n_children[p]++;
    }

    /* Cumulative offsets. */
    int acc = 0;
    for (int i = 0; i < nc; ++i) {
        V->child_start[i] = acc;
        acc += V->n_children[i];
    }
    V->child_start[nc] = acc;

    /* Fill: walk i again, append i to its parent's slot. */
    int cursor[SP_KSTE_MAX_NODES];
    memset(cursor, 0, sizeof(cursor));
    for (int i = 1; i < nc; ++i) {
        int p = sp_kste_unpack_parent(T->parents, i);
        if (p < 0 || p >= nc) continue;
        int slot = V->child_start[p] + cursor[p]++;
        V->child_idx[slot] = i;
    }
}

/* DFS pre-order over the children-in-stored-order; compute pre, size,
 * pre_inv.  Iterative with an explicit stack so we never blow the C
 * stack on pathological inputs. */
static void sp_kste_build_preorder(sp_kste_view *V)
{
    int n = V->node_count;
    memset(V->pre,     0, sizeof(V->pre));
    memset(V->size,    0, sizeof(V->size));
    memset(V->pre_inv, 0, sizeof(V->pre_inv));
    if (n == 0) return;

    /* Stack entry: (node, child_cursor) — we process this node's children
     * in order, descending one at a time.  Stack capacity = max tree depth. */
    typedef struct { int node, cursor; } frame_t;
    frame_t stack[SP_KSTE_MAX_NODES + 1];
    int sp = 0;
    int pre_ix = 0;

    /* Visit node 0 (root). */
    V->pre[0]     = pre_ix;
    V->pre_inv[pre_ix] = 0;
    pre_ix++;
    stack[sp++] = (frame_t){ 0, 0 };

    while (sp > 0) {
        frame_t *f = &stack[sp - 1];
        int nc = V->n_children[f->node];
        if (f->cursor < nc) {
            int child = V->child_idx[V->child_start[f->node] + f->cursor];
            f->cursor++;
            /* Visit child. */
            V->pre[child]      = pre_ix;
            V->pre_inv[pre_ix] = child;
            pre_ix++;
            stack[sp++] = (frame_t){ child, 0 };
        } else {
            /* All children processed; compute subtree size. */
            int sz = 1;
            for (int j = 0; j < nc; ++j) {
                int c = V->child_idx[V->child_start[f->node] + j];
                sz += V->size[c];
            }
            V->size[f->node] = sz;
            --sp;
        }
    }
}

static void sp_kste_build_view(const sp_kste_tree *T, sp_kste_view *V)
{
    sp_kste_build_children(T, V);
    sp_kste_build_preorder(V);
}

/* ---------- The decision procedure --------------------------------------- */

typedef struct {
    const sp_kste_tree *Qt;
    const sp_kste_tree *Kt;
    const sp_kste_view *Qv;
    const sp_kste_view *Kv;
    int depth;
    int max_depth;
    int steps;
    int backtracks;
    int capped;
} embed_state_t;

static int sp_kste_embed_subtree(embed_state_t *st, int q, int k);

static int sp_kste_embed_match_children(embed_state_t *st,
                                        int q_parent, int q_idx,
                                        int k_root, int pre_lo)
{
    /* Match q_parent's children starting at q_idx into descendants of
     * k_root with pre-order >= pre_lo. */
    const sp_kste_view *Q = st->Qv;
    const sp_kste_view *K = st->Kv;

    int q_nc = Q->n_children[q_parent];
    if (q_idx >= q_nc) return 1;  /* all matched */

    if (st->depth > SP_KSTE_EMBED_MAX_DEPTH) {
        st->capped = 1;
        return 1;
    }
    if (st->steps > SP_KSTE_EMBED_MAX_STEPS) {
        st->capped = 1;
        return 1;
    }

    int q_child = Q->child_idx[Q->child_start[q_parent] + q_idx];
    int k_pre_max = K->pre[k_root] + K->size[k_root] - 1;

    sp_kste_label q_lbl = sp_kste_unpack_label(st->Qt->labels, q_child);

    for (int p = pre_lo; p <= k_pre_max; ++p) {
        int k_prime = K->pre_inv[p];
        sp_kste_label k_lbl = sp_kste_unpack_label(st->Kt->labels, k_prime);
        if (k_lbl != q_lbl) continue;          /* fast label prefilter */

        if (!sp_kste_embed_subtree(st, q_child, k_prime)) {
            /* Candidate failed at the subtree level — try the next. */
            st->backtracks++;
            continue;
        }
        int next_pre = K->pre[k_prime] + K->size[k_prime];
        if (sp_kste_embed_match_children(st, q_parent, q_idx + 1,
                                         k_root, next_pre)) {
            return 1;
        }
        /* Subtree matched but downstream siblings failed — back out. */
        st->backtracks++;
    }
    return 0;
}

static int sp_kste_embed_subtree(embed_state_t *st, int q, int k)
{
    st->steps++;
    if (++st->depth > st->max_depth) st->max_depth = st->depth;

    if (st->depth > SP_KSTE_EMBED_MAX_DEPTH ||
        st->steps  > SP_KSTE_EMBED_MAX_STEPS) {
        st->capped = 1;
        --st->depth;
        return 1;   /* conservative-yes */
    }

    sp_kste_label q_lbl = sp_kste_unpack_label(st->Qt->labels, q);
    sp_kste_label k_lbl = sp_kste_unpack_label(st->Kt->labels, k);
    if (q_lbl != k_lbl) { --st->depth; return 0; }

    int q_nc = st->Qv->n_children[q];
    if (q_nc == 0) { --st->depth; return 1; }    /* Q-leaf — labels matched */

    int pre_lo = st->Kv->pre[k] + 1;             /* strict descendants */
    int ok = sp_kste_embed_match_children(st, q, 0, k, pre_lo);
    --st->depth;
    return ok;
}

/* ---------- Public API --------------------------------------------------- */

int sp_kste_embed_ex(const sp_kste_tree *Q, const sp_kste_tree *K,
                     sp_kste_embed_stats *stats)
{
    if (stats) {
        stats->backtracks = 0;
        stats->steps      = 0;
        stats->max_depth  = 0;
        stats->capped     = 0;
    }
    if (!Q || !K) return 0;
    if (Q->node_count <= 1) return 1;                 /* root-only Q       */
    if (Q->node_count > K->node_count) return 0;      /* size bound        */

    sp_kste_view Qv, Kv;
    sp_kste_build_view(Q, &Qv);
    sp_kste_build_view(K, &Kv);

    embed_state_t st;
    st.Qt = Q; st.Kt = K; st.Qv = &Qv; st.Kv = &Kv;
    st.depth = 0; st.max_depth = 0; st.steps = 0;
    st.backtracks = 0; st.capped = 0;

    int ok = sp_kste_embed_subtree(&st, 0, 0);

    if (stats) {
        stats->backtracks = st.backtracks;
        stats->steps      = st.steps;
        stats->max_depth  = st.max_depth;
        stats->capped     = st.capped;
    }
    return ok;
}

int sp_kste_embed(const sp_kste_tree *Q, const sp_kste_tree *K)
{
    return sp_kste_embed_ex(Q, K, NULL);
}

/* ============================================================ */
/* Path C — UNORDERED tree embedding (Phase 4b remediation).    */
/* ============================================================ */

/* Decision procedure: Q ⪯_u K iff there is an injection
 *   iota : V(Q) -> V(K)
 * such that
 *   - label(iota(v)) = label(v)
 *   - u ancestor of v in Q  ⇒  iota(u) ancestor of iota(v) in K
 * (sibling-order preservation REMOVED).
 *
 * Algorithm: depth-first matching with a per-K-node "claimed" bitmap.
 * For each Q-node q at recursion depth d, iterate K-descendants of
 * the parent's image and pick the first non-claimed candidate whose
 * label matches and whose subtree contains room for q's own children.
 * On success, mark the candidate's whole subtree claimed and recurse.
 * On failure, backtrack and try the next candidate.
 *
 * Bitmap is a `uint8_t[SP_KSTE_MAX_NODES]` — 60 bytes, stack-resident,
 * fast clone/restore for backtracking. */

typedef struct {
    const sp_kste_tree *Qt;
    const sp_kste_tree *Kt;
    const sp_kste_view *Qv;
    const sp_kste_view *Kv;
    uint8_t  claimed[SP_KSTE_MAX_NODES];   /* 1 = node already bound       */
    int      depth;
    int      max_depth;
    int      steps;
    int      backtracks;
    int      capped;
} embed_u_state_t;

/* Mark / unmark every node of K's subtree rooted at `k`. */
static void sp_kste_claim_subtree(embed_u_state_t *st, int k, uint8_t v)
{
    int n = st->Kv->size[k];
    int p0 = st->Kv->pre[k];
    for (int i = 0; i < n; ++i) {
        st->claimed[st->Kv->pre_inv[p0 + i]] = v;
    }
}

static int sp_kste_embed_u_subtree(embed_u_state_t *st, int q, int k);

static int sp_kste_embed_u_match_children(embed_u_state_t *st,
                                          int q_parent, int q_idx,
                                          int k_root)
{
    const sp_kste_view *Q = st->Qv;
    const sp_kste_view *K = st->Kv;

    if (q_idx >= Q->n_children[q_parent]) return 1;   /* all bound */

    if (st->depth > SP_KSTE_EMBED_MAX_DEPTH) { st->capped = 1; return 1; }
    if (st->steps > SP_KSTE_EMBED_MAX_STEPS) { st->capped = 1; return 1; }

    int q_child = Q->child_idx[Q->child_start[q_parent] + q_idx];
    sp_kste_label q_lbl = sp_kste_unpack_label(st->Qt->labels, q_child);

    int k_pre_max = K->pre[k_root] + K->size[k_root] - 1;
    /* For unordered semantics we iterate ALL non-claimed descendants of
     * k_root, in pre-order (deterministic; arbitrary order is fine). */
    for (int p = K->pre[k_root] + 1; p <= k_pre_max; ++p) {
        int k_prime = K->pre_inv[p];
        if (st->claimed[k_prime]) continue;
        sp_kste_label k_lbl = sp_kste_unpack_label(st->Kt->labels, k_prime);
        if (k_lbl != q_lbl) continue;

        /* Tentatively claim k_prime's whole subtree. */
        sp_kste_claim_subtree(st, k_prime, 1);
        if (sp_kste_embed_u_subtree(st, q_child, k_prime)) {
            if (sp_kste_embed_u_match_children(st, q_parent, q_idx + 1,
                                               k_root)) {
                return 1;
            }
        }
        /* Backtrack: release the claim. */
        sp_kste_claim_subtree(st, k_prime, 0);
        st->backtracks++;
    }
    return 0;
}

static int sp_kste_embed_u_subtree(embed_u_state_t *st, int q, int k)
{
    st->steps++;
    if (++st->depth > st->max_depth) st->max_depth = st->depth;
    if (st->depth > SP_KSTE_EMBED_MAX_DEPTH ||
        st->steps  > SP_KSTE_EMBED_MAX_STEPS) {
        st->capped = 1; --st->depth; return 1;
    }

    sp_kste_label q_lbl = sp_kste_unpack_label(st->Qt->labels, q);
    sp_kste_label k_lbl = sp_kste_unpack_label(st->Kt->labels, k);
    if (q_lbl != k_lbl) { --st->depth; return 0; }

    int q_nc = st->Qv->n_children[q];
    if (q_nc == 0) { --st->depth; return 1; }

    int ok = sp_kste_embed_u_match_children(st, q, 0, k);
    --st->depth;
    return ok;
}

int sp_kste_embed_unordered_ex(const sp_kste_tree *Q, const sp_kste_tree *K,
                               sp_kste_embed_stats *stats)
{
    if (stats) {
        stats->backtracks = 0; stats->steps = 0;
        stats->max_depth  = 0; stats->capped = 0;
    }
    if (!Q || !K) return 0;
    if (Q->node_count <= 1) return 1;
    if (Q->node_count > K->node_count) return 0;

    sp_kste_view Qv, Kv;
    sp_kste_build_view(Q, &Qv);
    sp_kste_build_view(K, &Kv);

    embed_u_state_t st;
    st.Qt = Q; st.Kt = K; st.Qv = &Qv; st.Kv = &Kv;
    memset(st.claimed, 0, sizeof(st.claimed));
    st.claimed[0] = 1;   /* root maps to root */
    st.depth = 0; st.max_depth = 0; st.steps = 0;
    st.backtracks = 0; st.capped = 0;

    int ok = sp_kste_embed_u_subtree(&st, 0, 0);

    if (stats) {
        stats->backtracks = st.backtracks;
        stats->steps      = st.steps;
        stats->max_depth  = st.max_depth;
        stats->capped     = st.capped;
    }
    return ok;
}

int sp_kste_embed_unordered(const sp_kste_tree *Q, const sp_kste_tree *K)
{
    return sp_kste_embed_unordered_ex(Q, K, NULL);
}
