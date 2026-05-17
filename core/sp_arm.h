/* sp_arm.h — Shannon-Prime Algebraic Resonance Memory (Phase 13.A).
 *
 * Holographic Reduced Representation style associative memory native to
 * the dual-prime CRT cyclotomic ring R_q1 × R_q2 (q1, q2 30-bit Proth,
 * N = SP_NTT_CRT_N = 256).
 *
 * Math:
 *   STORE (k_i, v_i):   M += NTT(encode(k_i)) ⊙ NTT(encode(v_i))
 *   RECALL (q):         out = INTT( M ⊙ NTT(encode_reversed(q)) )
 *                       ≈ v_j   when q ≈ k_j   (modulo cross-talk noise)
 *
 * Both operations are O(N) pointwise multiplies per CRT shard — same
 * primitives as sp_poly_dot_product_ntt_crt_qk_cached. The only new
 * kernel here is the full-coefficient inverse-NTT + CRT-stitch + fp32
 * decode (the dot-product helper extracts a single coefficient).
 *
 * No floats inside the bank: storage and superposition happen entirely
 * in Z_q[x]/(x^N+1) integer coefficients. fp32 only at the encode and
 * decode boundaries.
 *
 * Capacity (rough): classical HRR ≈ N / (4 log N) ≈ 8 reliable patterns
 * per slab at N=256. Our 30-bit prime modulus gives extra headroom;
 * expect ~16-32 usable patterns per slab. Many slabs allowed (a slab
 * is just N coefficients × 2 primes × 8 bytes = 4 KB).
 */

#ifndef SP_ARM_H
#define SP_ARM_H

#include <stdint.h>
#include <stddef.h>

#include "sp_ntt_crt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Ring degree (fixed at SP_NTT_CRT_N = 256 for the dual-prime NTT). */
#define SP_ARM_RING_N SP_NTT_CRT_N

/* Algebraic Resonance Memory bank.
 *
 * `M_q1` and `M_q2` are caller-owned buffers of size n_slabs * N each.
 * Each slab is one independent memory pattern accumulator.
 *
 * `d` is the active dimension — k_vec / v_vec / q_vec are length-d
 * fp32 vectors; the rest of the N-coefficient ring is zero-padded by
 * the encoder. d must be ≤ N. Typical use: d = head_dim = 256.
 *
 * `delta` is the integer scale at encode time. Smaller delta = less
 * dynamic range per coefficient but more headroom for accumulation.
 * Recommended starting value: 1<<8 (256). Tune empirically via
 * test_sp_arm capacity sweep.
 */
typedef struct sp_arm_bank {
    uint64_t* M_q1;
    uint64_t* M_q2;
    int       n_slabs;
    int       N;            /* fixed at SP_ARM_RING_N */
    int       d;            /* active dimension, d <= N */
    double    delta;        /* integer encode scale */
    int       n_writes;     /* total writes across all slabs */
} sp_arm_bank;

/* Initialize a bank.  M_q1 / M_q2 must be n_slabs * N uint64s.  The
 * accumulators are zeroed. */
void sp_arm_bank_init(sp_arm_bank* bank,
                       uint64_t* M_q1, uint64_t* M_q2,
                       int n_slabs, int d, double delta);

/* Zero one slab's accumulator. */
void sp_arm_bank_clear_slab(sp_arm_bank* bank, int slab);

/* Bind (k, v) and accumulate into slab.
 *
 *   M_q[slab][i] = ( M_q[slab][i] + K_q[i] * V_q[i] ) mod q
 *
 * for each i, each prime q in {q1, q2}, where K_q / V_q are forward-NTT
 * representations of encode_fp32(k_vec, delta) / encode_fp32(v_vec, delta).
 *
 * `scratch_4N` is a caller-supplied scratch buffer of length 4 * N
 * uint64s (two NTT lanes for K plus two for V). Plus a separate
 * length-N int64 buffer `int_scratch` for the encode-coefficient step.
 *
 * No floats in the inner path beyond the encode.
 */
void sp_arm_bank_write(sp_arm_bank* bank, int slab,
                        const float* k_vec, const float* v_vec,
                        uint64_t* scratch_4N,
                        int64_t* int_scratch);

/* Recall: compute v_out ≈ v_j when q_vec ≈ k_j.
 *
 * Math identity (HRR-style in the negacyclic ring R = Z[x]/(x^N+1)):
 *
 *   M(x) = sum_i  K_i(x) * V_i(x)
 *   M(x) * inv(q)(x) ≈ ||q||^2 * V_j(x)  +  cross-talk
 *
 * where inv(q) is the negacyclic involution:
 *   inv(q)(x) = q(x^{-1}) = q(-x^{N-1})
 *
 * In coefficient form (with d ≤ N and q zero-padded to length N):
 *   inv_q[0]      =  q[0]
 *   inv_q[N-j]    = -q[j]     for j in [1, d-1]
 *   inv_q[i]      =  0        elsewhere
 *
 * Pipeline:
 *   inv_q_fp32 = involution(q_vec)
 *   Inv_q      = NTT(encode(inv_q_fp32, delta))
 *   Out_q[i]   = M_q[slab][i] * Inv_q[i] mod q   (pointwise per prime)
 *   coeffs     = INTT(Out_q)                      (back to coeff domain)
 *   v_out[i]   = coeffs[i] / (delta^3)            (fp32 decode)
 *
 * Decode by delta^3 because K, V, and inv(q) each contributed one
 * factor of delta at encode time.
 *
 * `scratch_4N` is length 4*N uint64s.
 * `inv_q_fp32_scratch` is length N float — caller-provided workspace
 * for the involution computation.
 */
void sp_arm_bank_recall(const sp_arm_bank* bank, int slab,
                         const float* q_vec,
                         float* v_out_fp32,
                         uint64_t* scratch_4N,
                         int64_t* int_scratch,
                         float* inv_q_fp32_scratch);

/* Compute the negacyclic involution of a length-d vector, padded to N.
 * Public for testing — the recall path uses this internally.
 *   inv_out[0]      =  in[0]
 *   inv_out[N-j]    = -in[j]   for j in [1, d-1]
 *   inv_out[i]      =  0       elsewhere
 */
void sp_arm_involution_fp32(const float* in, int d, int N, float* inv_out);

/* Sum-of-squares of the slab's coefficients after inverse NTT.  Used
 * as a soft "memory mass" / threshold check.
 *
 *   ||M||^2 = sum_{i=0..N-1} (INTT(M)_i)^2
 *
 * Computed in fp64 after CRT-stitching.  `scratch_2N` is length
 * 2*N uint64s.
 */
double sp_arm_bank_norm(const sp_arm_bank* bank, int slab,
                         uint64_t* scratch_2N);

#ifdef __cplusplus
}
#endif

#endif /* SP_ARM_H */
