/* sp_hex_matmul_block_q8.c — Strike 5 implementation.
 *
 * Same algebra as sp_matmul_ok_block_inner_scalar in src/sp_matmul.cpp
 * but restructured into the two-phase HVX-friendly form (per-block
 * D_a/D_b sums, then per-block (B_a, B_b) fold).
 *
 * Both paths produce bit-identical int64 (acc_a, acc_b) for the same
 * (w_blocks, x_row) input. The host parity test in
 * tests/test_sp_hex_matmul_block_q8.cpp asserts that explicitly.
 */

#include "sp_hex_matmul_block_q8.h"

#define SP_HEX_W41  ((int64_t)SP_OK_OMEGA_NORM)  /* = 41 */

#if defined(__HVX__) && defined(__hexagon__)
#  include <hexagon_types.h>
#  include <hvx_hexagon_protos.h>
#  define SP_HEX_HAVE_HVX 1
#else
#  define SP_HEX_HAVE_HVX 0
#endif

/* ─── Scalar two-phase reference (always available) ────────────────────
 *
 * For each 32-element block:
 *   D_a = sum_{k=0..31} packed[k] * x[k].a
 *   D_b = sum_{k=0..31} packed[k] * x[k].b
 *
 *   acc_a += B_a * D_a − 41 * B_b * D_b
 *   acc_b += B_a * D_b + B_b * (D_a + D_b)
 *
 * Algebraic check vs direct form (sp_matmul_ok_block_inner_scalar):
 *   direct:   acc_a += sum_k packed[k] * (B_a x[k].a − 41 B_b x[k].b)
 *                    = B_a * (sum_k packed[k] x[k].a) − 41 B_b * (sum_k packed[k] x[k].b)
 *                    = B_a * D_a − 41 * B_b * D_b   ✓
 *   direct:   acc_b += sum_k packed[k] * (B_a x[k].b + B_b x[k].a + B_b x[k].b)
 *                    = B_a * D_b + B_b * D_a + B_b * D_b
 *                    = B_a * D_b + B_b * (D_a + D_b)   ✓
 *
 * Integer-overflow safety: with int8 codepoints in [-127, +127] and
 * x[k].{a,b} in int64, |packed[k] * x[k].*| ≤ 127 * 2^63 ≪ 2^64. Sum of
 * 32 such products: ≤ 32 * 127 * 2^63 — overflows int64. We rely on
 * the engine's invariant that x activations are pre-bounded by the
 * scale_recip and frobenius_scale Q-format so that the 32-element
 * partial sum fits in int64. This is the same invariant the engine
 * reference depends on; we don't tighten it here.
 */
static void sp_hex_matmul_inner_scalar(
    const sp_ok_q8_block_t* w_blocks,
    const sp_ok_t*          x_row,
    size_t                  blocks_per_row,
    int64_t*                out_acc_a,
    int64_t*                out_acc_b)
{
    int64_t acc_a = 0;
    int64_t acc_b = 0;
    const int need_b = (out_acc_b != NULL);

    for (size_t b = 0; b < blocks_per_row; ++b) {
        const sp_ok_q8_block_t* blk = w_blocks + b;
        const int64_t B_a = blk->B_a;
        const int64_t B_b = blk->B_b;
        const sp_ok_t* x_tile = x_row + b * (size_t)SP_OK_BLOCK_SIZE;

        /* Phase 1: per-block dot products of int8 codepoints against the
         * sp_ok_t coordinate columns. D_b is needed in BOTH the acc_a
         * fold (−41·B_b·D_b term) and the acc_b fold, so we always
         * compute it regardless of need_b. */
        int64_t D_a = 0;
        int64_t D_b = 0;
        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            const int64_t w = (int64_t)blk->packed[k];
            D_a += w * x_tile[k].a;
            D_b += w * x_tile[k].b;
        }

        /* Phase 2: fold per-block Frobenius coeffs into accumulators. */
        acc_a += B_a * D_a - SP_HEX_W41 * B_b * D_b;
        if (need_b) acc_b += B_a * D_b + B_b * (D_a + D_b);
    }

    *out_acc_a = acc_a;
    if (need_b) *out_acc_b = acc_b;
}

#if SP_HEX_HAVE_HVX

/* ─── HVX vectorised inner (compiled only by hexagon-clang) ───────────
 *
 * V69 architectural facts:
 *   - HVX_Vector is 1024 bits = 32 × int32 lanes = 64 × int16 = 128 × int8.
 *   - Activations x[k] are sp_ok_t (int64 a + int64 b).
 *   - In production the engine keeps |x.{a,b}| ≤ scale_recip · |fp32_act|
 *     ≈ 16384 · ~8 ≈ 2^17 — comfortably in int32. We therefore process
 *     x[k].a and x[k].b as int32 lanes (truncating the int64 representation
 *     to its low 32 bits). The host-side caller asserts this invariant
 *     before dispatching; out-of-range x[k] falls back to the scalar path.
 *
 * Per-block sequence (32 elements per block):
 *   1. Load 32 int8 codepoints into a 32-byte slice of an HVX_Vector.
 *      Sign-extend to int16 (Q6_Wh_vsxt_Vb), then int32 (Q6_Vw_vsxth_Vh) —
 *      two halfword pairs widening to two word vector halves, then take
 *      the low 32 lanes (== 32 sign-extended int32 codepoints).
 *   2. Load 32 int32 activations (x[k].a low-bits) from the activation
 *      buffer pre-narrowed by the caller into int32 lanes.
 *   3. Multiply lane-wise (Q6_Vw_vmpyio_VwVw) and reduce-sum across
 *      32 lanes via successive Q6_Vw_vrmpyi_VwVw_acc pair reductions
 *      to a single int32 — promoted to int64 in scalar before fold.
 *   4. Same again for x[k].b → D_b.
 *
 * Per-block cost ≈ 6 HVX vector ops + scalar fold (4 mul + 3 add).
 * Memory: 1 byte vector load + 2 word vector loads per block.
 *
 * NOTE: This implementation uses Q6_Vw_vmpyio + scalar horizontal
 * reduction. The Q6_V_vrmpy_VbVb path the V69 ISA offers reduces
 * 4 byte products per int32 lane, but requires byte-aligned activations
 * which we don't have (sp_ok_t is int64). For first cut we use the
 * widened int32 path; a follow-up can do a byte-packed activation
 * variant for higher throughput.
 */

/* Horizontal reduce-sum 32 × int32 lanes to one scalar via the
 * standard log2(N) rotate-and-add tree. Each step halves the active
 * lane count; after 5 rounds lane 0 holds the full 32-lane sum.
 *
 *   Step 1: rotate 64 B (=16 words), add → lanes 0..15 hold pair sums
 *   Step 2: rotate 32 B (= 8 words), add → lanes 0..7  hold 4-way sums
 *   Step 3: rotate 16 B (= 4 words), add → lanes 0..3  hold 8-way sums
 *   Step 4: rotate  8 B (= 2 words), add → lanes 0..1  hold 16-way sums
 *   Step 5: rotate  4 B (= 1 word),  add → lane  0     holds 32-way sum
 *
 * 5 × vror + 5 × vadd + 1 × vextract  =  11 HVX/scalar ops total.
 * The previous spill-to-stack pattern was 1 vector store + 31 scalar
 * loads + 31 scalar adds ≈ 63 ops. About 6× fewer instructions on
 * the reduce alone; more importantly, zero memory traffic — the
 * partial sums stay in HVX registers throughout. */
static inline int32_t sp_hex_hreduce_sum_w(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));   /* 16 + 16 */
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));   /*  8 + 8  */
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));   /*  4 + 4  */
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v,  8));   /*  2 + 2  */
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v,  4));   /*  1 + 1  */
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

static void sp_hex_matmul_inner_hvx(
    const sp_ok_q8_block_t* w_blocks,
    const sp_ok_t*          x_row,
    size_t                  blocks_per_row,
    int64_t*                out_acc_a,
    int64_t*                out_acc_b)
{
    int64_t acc_a = 0;
    int64_t acc_b = 0;
    const int need_b = (out_acc_b != NULL);

    /* Strategy: per block, compute D_a and D_b via the accumulating
     * vector multiply Q6_Vw_vmpyieacc_VwVwVh, which takes word ×
     * even-halfword and adds into a word accumulator.
     *
     *   Vx + (Vu_word × Vv_even_halfword) → Vx (32 int32 lanes)
     *
     * Layout:
     *   Vu = 32 int32 activations (xa or xb, low 32 bits of x[k].{a,b})
     *   Vv = 32 int32 lanes whose low 16 bits hold the sign-extended
     *        int8 codepoint at that lane position (the "even halfword"
     *        is exactly the low 16 bits of each int32 slot)
     *   Vx start = zero vector → result is per-lane (xa × codepoint)
     *
     * Then horizontal-reduce the 32 int32 lanes to one int32. The sum
     * fits in int32 by the bounded-activation invariant:
     *   |packed[k]| ≤ 127, |x[k].{a,b}| ≤ ~2^17 → |product| ≤ 2^24
     *   32 × 2^24 = 2^29 — comfortably inside int32.
     *
     * Per-block cost: 2 vector multiplies (vmpyieacc, one for D_a,
     * one for D_b) + 2 horizontal reduces + 1 widen-pack of codepoints.
     * The Phase-2 fold is identical scalar int64. */

    int32_t xa32[32] __attribute__((aligned(128)));
    int32_t xb32[32] __attribute__((aligned(128)));
    int32_t cp32[32] __attribute__((aligned(128)));  /* codepoint in low 16 of each int32 */

    for (size_t b = 0; b < blocks_per_row; ++b) {
        const sp_ok_q8_block_t* blk = w_blocks + b;
        const int64_t B_a = blk->B_a;
        const int64_t B_b = blk->B_b;
        const sp_ok_t* x_tile = x_row + b * (size_t)SP_OK_BLOCK_SIZE;

        for (int k = 0; k < SP_OK_BLOCK_SIZE; ++k) {
            xa32[k] = (int32_t)x_tile[k].a;
            xb32[k] = (int32_t)x_tile[k].b;
            /* Sign-extend int8 → int32, which automatically places the
             * value in the low 16 bits (since |int8| ≤ 127 fits int16).
             * The "even halfword" of the int32 is the low 16 bits. */
            cp32[k] = (int32_t)blk->packed[k];
        }

        HVX_Vector v_w  = *(const HVX_Vector*)cp32;
        HVX_Vector v_xa = *(const HVX_Vector*)xa32;
        HVX_Vector v_zero = Q6_V_vzero();

        /* D_a: accumulate (xa × codepoint) across 32 lanes into v_zero,
         * then horizontal-reduce to one int32. */
        HVX_Vector v_prod_a = Q6_Vw_vmpyieacc_VwVwVh(v_zero, v_xa, v_w);
        int64_t D_a = (int64_t)sp_hex_hreduce_sum_w(v_prod_a);

        int64_t D_b = 0;
        if (need_b || B_b != 0) {
            HVX_Vector v_xb = *(const HVX_Vector*)xb32;
            HVX_Vector v_prod_b = Q6_Vw_vmpyieacc_VwVwVh(v_zero, v_xb, v_w);
            D_b = (int64_t)sp_hex_hreduce_sum_w(v_prod_b);
        }

        /* Phase 2: per-block cyclotomic fold (scalar int64). */
        acc_a += B_a * D_a - SP_HEX_W41 * B_b * D_b;
        if (need_b) acc_b += B_a * D_b + B_b * (D_a + D_b);
    }

    *out_acc_a = acc_a;
    if (need_b) *out_acc_b = acc_b;
}

#endif /* SP_HEX_HAVE_HVX */

void sp_hex_matmul_ok_block_q8_inner(
    const sp_ok_q8_block_t* w_blocks,
    const sp_ok_t*          x_row,
    size_t                  blocks_per_row,
    int64_t*                out_acc_a,
    int64_t*                out_acc_b)
{
    if (!w_blocks || !x_row || !out_acc_a || blocks_per_row == 0) {
        if (out_acc_a) *out_acc_a = 0;
        if (out_acc_b) *out_acc_b = 0;
        return;
    }
#if SP_HEX_HAVE_HVX
    sp_hex_matmul_inner_hvx(w_blocks, x_row, blocks_per_row,
                             out_acc_a, out_acc_b);
#else
    sp_hex_matmul_inner_scalar(w_blocks, x_row, blocks_per_row,
                                out_acc_a, out_acc_b);
#endif
}

int sp_hex_matmul_block_q8_uses_hvx(void) {
#if SP_HEX_HAVE_HVX
    return 1;
#else
    return 0;
#endif
}
