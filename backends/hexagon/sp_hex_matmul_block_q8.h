/* sp_hex_matmul_block_q8.h — Strike 5: HVX cyclotomic matmul kernel.
 *
 * One inner-row primitive for matmul against sp_ok_q8_block_t-packed
 * weights. Designed so the math is structurally HVX-friendly: it splits
 * the per-element B_a/B_b mix into two phases so the inner loop becomes
 * a pure int8 × int64 dot product per block (which HVX vectorises via
 * Q6_V_vrmpy / Q6_Vw_vmpa byte-product reductions).
 *
 * Math equivalence with the engine reference (sp_matmul.cpp's
 * sp_matmul_ok_block_inner_scalar):
 *
 *   Direct form (engine reference):
 *     acc_a += packed[k] * (B_a * x[k].a − 41 * B_b * x[k].b)
 *     acc_b += packed[k] * (B_a * x[k].b + B_b * x[k].a + B_b * x[k].b)
 *
 *   HVX-friendly two-phase form:
 *     D_a(b) = sum_{k in block} packed[k] * x[k].a
 *     D_b(b) = sum_{k in block} packed[k] * x[k].b
 *     acc_a += B_a(b) * D_a(b) − 41 * B_b(b) * D_b(b)
 *     acc_b += B_a(b) * D_b(b) + B_b(b) * (D_a(b) + D_b(b))
 *
 * The two forms are algebraically identical — by distributivity of the
 * inner sum across the per-block (B_a, B_b). The two-phase form lets
 * HVX issue 32 byte-dot-product MACs in one Q6_V_vrmpy_VbVb against the
 * low/high halves of x[k].a and x[k].b separately, then a small per-
 * block scalar fold combines them with the block's Frobenius coeffs.
 *
 * Build path:
 *   - Host (this file): scalar reference, validates the two-phase
 *     restructuring is bit-equivalent to the direct form. Linked into
 *     the engine's tests.
 *   - Hexagon (Strike 6): same function name, real HVX intrinsics under
 *     #ifdef __HVX__. Same callable shape; FastRPC stub passes
 *     pre-fused weight blocks + activation row, returns acc_a/acc_b.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#ifndef SP_HEX_MATMUL_BLOCK_Q8_H
#define SP_HEX_MATMUL_BLOCK_Q8_H

#include "sp_ok_arith.h"
#include "sp_ok_block_quant.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Inner-row matmul kernel. Compute one (i, j) output coefficient of
 *   Y[i, j] = sum_k W[i, k] · X[j, k]
 * where W's row i is given as `w_blocks` (blocks_per_row × 64 B each)
 * and X's row j is `x_row` (blocks_per_row × 32 sp_ok_t elements).
 *
 * out_acc_a / out_acc_b receive the O_K-ring accumulator pair.
 * out_acc_b may be NULL — caller passes NULL when only acc_a is
 * needed (matches the engine's A_ONLY=true template path used by the
 * fp32-output matmul). */
void sp_hex_matmul_ok_block_q8_inner(
    const sp_ok_q8_block_t* w_blocks,
    const sp_ok_t*          x_row,
    size_t                  blocks_per_row,
    int64_t*                out_acc_a,
    int64_t*                out_acc_b);

/* Returns 1 if this build was compiled with __HVX__ defined and uses
 * the vector intrinsics; 0 if the scalar fallback ran. */
int sp_hex_matmul_block_q8_uses_hvx(void);

#ifdef __cplusplus
}
#endif

#endif /* SP_HEX_MATMUL_BLOCK_Q8_H */
