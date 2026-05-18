/* sp_hex_mobius_scatter.h — Strike 6: HVX Möbius reorder via vscatter.
 *
 * Moves sp_mobius_reorder from the A710 NEON scalar loop to the V69 HVX
 * scatter unit. Per 32-element chunk:
 *   1. Load 32 fp32 source coefficients into Vw
 *   2. Load 32 byte offsets from the compile-time table into Vv
 *   3. Q6_vscatter_RMVwV scatters all 32 lanes to VTCM in one issue
 * After all chunks, a vmem load from the VTCM region serves as the
 * scatter_release barrier (the load blocks until pending scatters drain).
 *
 * Numerical contract: out_reordered must be bit-equal to the host's
 * sp_mobius_reorder_ex(in_coeffs, mask) for the same head_dim. We're
 * just moving bytes via a different mechanism, no math involved.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3 / commercial.
 */

#ifndef SP_HEX_MOBIUS_SCATTER_H
#define SP_HEX_MOBIUS_SCATTER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reorder in_coeffs[0..head_dim) into out_reordered using HVX vscatter
 * via the session's VTCM scratch region.
 *
 *   in_coeffs    : aligned fp32 source, length head_dim
 *   head_dim     : must be in {64, 128, 256, 512} (else returns -1)
 *   out_reordered: aligned fp32 destination, length head_dim
 *   vtcm_scratch : pointer to VTCM region (acquired at session open)
 *   vtcm_bytes   : size of VTCM region in bytes (must be >= head_dim * 4)
 *
 * Returns 0 on success, -1 on shape mismatch or unsupported head_dim.
 *
 * Build path:
 *   __HVX__ defined → real Q6_vscatter intrinsics
 *   otherwise       → portable scalar fallback (writes to out_reordered
 *                     directly via the inverse permutation), same
 *                     numerical contract. */
int sp_hex_mobius_scatter_f32_dsp(const float* in_coeffs,
                                  int          head_dim,
                                  float*       out_reordered,
                                  void*        vtcm_scratch,
                                  size_t       vtcm_bytes);

/* Build-time discovery. */
int sp_hex_mobius_scatter_uses_hvx(void);

#ifdef __cplusplus
}
#endif

#endif /* SP_HEX_MOBIUS_SCATTER_H */
