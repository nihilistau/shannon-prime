/* sp_kste_choice.c — Phase 7 / Paper IV §10 Choice Operator F.
 *
 * Picks the canonical representative of a ⪯_d-equivalence class of
 * sp_kste_tree under the packed-lexicographic order on the 64-byte
 * representation.  Deterministic, order-invariant, O(n) comparisons
 * × O(64) bytes each.
 *
 * This file is the system-side implementation of the axiomatic layer
 * described in Paper III §11 (Choice Operator + Extended-Domain
 * Reduction).  See sp_kste.h for the public API.
 *
 * Invariants (verified by tests/test_sp_ultraproduct_attn.cpp):
 *   - sp_kste_select_canonical(trees, n) returns the same content
 *     regardless of permutation of `trees`.
 *   - Stable under content-equivalent re-encodings: two trees with
 *     bit-identical packed bytes compare equal.
 *   - sp_kste_tree_compare is a strict total order on the image of
 *     valid sp_kste_tree (since the packed bytes are deterministic
 *     after sp_kste_tree_clear + add_child sequence).
 */

#include "sp_kste.h"

#include <stddef.h>
#include <string.h>

int sp_kste_tree_compare(const sp_kste_tree *a, const sp_kste_tree *b)
{
    if (a == b) return 0;
    if (!a)     return -1;
    if (!b)     return  1;
    return memcmp(a, b, sizeof(sp_kste_tree));
}

const sp_kste_tree*
sp_kste_select_canonical(const sp_kste_tree * const *trees, int n)
{
    if (!trees || n <= 0) return NULL;

    /* Walk the array, keeping the lex-min seen so far. */
    const sp_kste_tree *best = NULL;
    for (int i = 0; i < n; ++i) {
        const sp_kste_tree *t = trees[i];
        if (!t) continue;
        if (best == NULL || sp_kste_tree_compare(t, best) < 0) {
            best = t;
        }
    }
    return best;
}
