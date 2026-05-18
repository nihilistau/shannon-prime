/* sp_hex_mobius_scatter.c — Strike 6 implementation.
 *
 * HVX path uses Q6_vscatter_RMVwV (word scatter) writing into the
 * session's VTCM region, then reads back via vmem to drain the
 * scatter queue (the load is the scatter_release barrier — hardware
 * stalls it until pending scatters commit).
 *
 * The non-HVX host path is a scalar reference that produces the SAME
 * numerical result by applying the inverse permutation directly. The
 * Strike 6 parity test compares the two paths bit-equal for head_dims
 * 64/128/256/512.
 */

#include "sp_hex_mobius_scatter.h"
#include "sp_hex_mobius_tables.h"

#include <stdint.h>
#include <string.h>

#if defined(__HVX__) && defined(__hexagon__)
#  include <hexagon_types.h>
#  include <hvx_hexagon_protos.h>
#  define SP_HEX_HAVE_HVX_SCATTER 1
#else
#  define SP_HEX_HAVE_HVX_SCATTER 0
#endif

int sp_hex_mobius_scatter_uses_hvx(void) {
#if SP_HEX_HAVE_HVX_SCATTER
    return 1;
#else
    return 0;
#endif
}

/* Common shape checks and table lookup, runs on both paths. */
static int sp_hex_mobius_validate(int head_dim, size_t vtcm_bytes,
                                   const uint32_t** out_offsets) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        return -1;
    }
    if ((size_t)head_dim * sizeof(float) > vtcm_bytes) {
        /* Caller didn't acquire enough VTCM for our region. */
        return -1;
    }
    const uint32_t* offsets = sp_hex_mobius_offsets_f32(head_dim);
    if (!offsets) return -1;
    *out_offsets = offsets;
    return 0;
}

int sp_hex_mobius_scatter_f32_dsp(const float* in_coeffs,
                                  int          head_dim,
                                  float*       out_reordered,
                                  void*        vtcm_scratch,
                                  size_t       vtcm_bytes)
{
    /* out_reordered == NULL is the "fused chain" mode: the scattered
     * data stays in VTCM for the next DSP kernel to consume directly,
     * skipping the DDR round-trip. The next kernel's vmem load from
     * VTCM serves as the scatter_release barrier. */
    if (!in_coeffs || !vtcm_scratch) return -1;
    const uint32_t* offsets = NULL;
    if (sp_hex_mobius_validate(head_dim, vtcm_bytes, &offsets) != 0) return -1;

#if SP_HEX_HAVE_HVX_SCATTER
    /* HVX path: vscatter into VTCM, then vmem-load back into out_reordered.
     *
     * Each scatter handles 32 word lanes per issue. We chunk head_dim by
     * 32 — for hd=128 that's 4 scatter issues + 4 vmem loads. The base
     * pointer Rt and modifier Mu (region_size − 1) stay constant; only
     * Vv (offsets) and Vw (data) advance per chunk. */
    /* Hexagon scalar register args are 32-bit. Cast pointer through
     * `unsigned long` (always visible, ≥32-bit) then narrow to
     * `unsigned int` which matches the intrinsic's Word32. */
    const unsigned int Rt = (unsigned int)(unsigned long)vtcm_scratch;
    const unsigned int Mu = (unsigned int)((head_dim * sizeof(float)) - 1u);

    /* Issue scatters in 32-lane chunks. */
    for (int k = 0; k < head_dim; k += 32) {
        HVX_Vector Vv = *((const HVX_Vector*)(offsets   + k));
        HVX_Vector Vw = *((const HVX_Vector*)(in_coeffs + k));
        Q6_vscatter_RMVwV(Rt, Mu, Vv, Vw);
    }

    /* Read back from VTCM into the output buffer ONLY if the caller
     * asked for it. In the fused-chain case (out_reordered == NULL),
     * the next DSP kernel reads directly from VTCM — its vmem load is
     * the scatter_release barrier and there's no DDR round-trip. */
    if (out_reordered != NULL) {
        const volatile HVX_Vector* vtcm_vp = (const volatile HVX_Vector*)vtcm_scratch;
        HVX_Vector* dst_vp = (HVX_Vector*)out_reordered;
        const int n_vec = head_dim / 32;
        for (int v = 0; v < n_vec; ++v) {
            dst_vp[v] = vtcm_vp[v];
        }
    }
    return 0;
#else
    /* Scalar reference path (host build / __HVX__-disabled).
     * Apply the inverse-permutation byte offsets directly. In fused
     * mode (out_reordered == NULL) we write to VTCM scratch instead. */
    (void)vtcm_bytes;
    float* dst = out_reordered ? out_reordered : (float*)vtcm_scratch;
    for (int j = 0; j < head_dim; ++j) {
        const uint32_t byte_off = offsets[j];
        const int dst_idx = (int)(byte_off / sizeof(float));
        dst[dst_idx] = in_coeffs[j];
    }
    return 0;
#endif
}
