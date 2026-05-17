/* sp_arm.c — Algebraic Resonance Memory implementation.
 *
 * Phase 13.A. See sp_arm.h for the math.
 *
 * Everything below builds on existing primitives:
 *   sp_poly_encode_ntt_q_crt           : fp32 -> forward NTT under (q1, q2)
 *   sp_poly_encode_ntt_k_reversed_crt  : fp32 -> reversed forward NTT
 *   sp_ntt_crt_pointwise_mul           : NTT-domain pointwise multiply
 *   sp_ntt_crt_inverse                 : inverse NTT back to coefficients
 *   sp_ntt_crt_combine                 : CRT stitch one coefficient
 *
 * No new kernel primitives are introduced; sp_arm just composes them
 * with the right buffer plumbing for the write/recall pattern.
 */

#include "sp_arm.h"
#include "sp_ntt_crt.h"

#include <string.h>

void sp_arm_bank_init(sp_arm_bank* bank,
                       uint64_t* M_q1, uint64_t* M_q2,
                       int n_slabs, int d, double delta) {
    bank->M_q1     = M_q1;
    bank->M_q2     = M_q2;
    bank->n_slabs  = n_slabs;
    bank->N        = SP_ARM_RING_N;
    bank->d        = d;
    bank->delta    = delta;
    bank->n_writes = 0;
    if (M_q1 != 0 && M_q2 != 0 && n_slabs > 0) {
        const size_t n = (size_t)n_slabs * (size_t)SP_ARM_RING_N;
        memset(M_q1, 0, n * sizeof(uint64_t));
        memset(M_q2, 0, n * sizeof(uint64_t));
    }
}

void sp_arm_bank_clear_slab(sp_arm_bank* bank, int slab) {
    if (slab < 0 || slab >= bank->n_slabs) return;
    const size_t off = (size_t)slab * (size_t)SP_ARM_RING_N;
    memset(bank->M_q1 + off, 0, SP_ARM_RING_N * sizeof(uint64_t));
    memset(bank->M_q2 + off, 0, SP_ARM_RING_N * sizeof(uint64_t));
}

void sp_arm_bank_write(sp_arm_bank* bank, int slab,
                        const float* k_vec, const float* v_vec,
                        uint64_t* scratch_4N,
                        int64_t* int_scratch) {
    if (slab < 0 || slab >= bank->n_slabs) return;
    if (!k_vec || !v_vec || !scratch_4N || !int_scratch) return;
    const int N = SP_ARM_RING_N;

    uint64_t* K_q1 = scratch_4N + 0 * N;
    uint64_t* K_q2 = scratch_4N + 1 * N;
    uint64_t* V_q1 = scratch_4N + 2 * N;
    uint64_t* V_q2 = scratch_4N + 3 * N;

    /* Forward NTT encode of K (forward order, not reversed). */
    sp_poly_encode_ntt_q_crt(K_q1, K_q2, k_vec, bank->d, bank->delta,
                              int_scratch);

    /* Forward NTT encode of V (also forward). */
    sp_poly_encode_ntt_q_crt(V_q1, V_q2, v_vec, bank->d, bank->delta,
                              int_scratch);

    /* Bind: produce K * V in NTT domain (pointwise) for each prime. */
    /* K_q1 := K_q1 ⊙ V_q1   (overwrite K_q1 to save a buffer) */
    sp_ntt_crt_pointwise_mul(K_q1, K_q1, V_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_pointwise_mul(K_q2, K_q2, V_q2, &SP_NTT_CRT_CTX_Q2);

    /* Accumulate into the bank slab. */
    uint64_t* slab_q1 = bank->M_q1 + (size_t)slab * (size_t)N;
    uint64_t* slab_q2 = bank->M_q2 + (size_t)slab * (size_t)N;
    const uint64_t q1 = SP_NTT_CRT_Q1;
    const uint64_t q2 = SP_NTT_CRT_Q2;
    for (int i = 0; i < N; ++i) {
        uint64_t s1 = slab_q1[i] + K_q1[i];
        if (s1 >= q1) s1 -= q1;
        slab_q1[i] = s1;
        uint64_t s2 = slab_q2[i] + K_q2[i];
        if (s2 >= q2) s2 -= q2;
        slab_q2[i] = s2;
    }
    bank->n_writes += 1;
}

void sp_arm_involution_fp32(const float* in, int d, int N, float* inv_out) {
    /* inv_out[0]    =  in[0]
     * inv_out[N-j]  = -in[j]   for j in [1, d-1]
     * inv_out[i]    =  0       elsewhere */
    for (int i = 0; i < N; ++i) inv_out[i] = 0.0f;
    inv_out[0] = in[0];
    for (int j = 1; j < d; ++j) {
        inv_out[N - j] = -in[j];
    }
}

void sp_arm_bank_recall(const sp_arm_bank* bank, int slab,
                         const float* q_vec,
                         float* v_out_fp32,
                         uint64_t* scratch_4N,
                         int64_t* int_scratch,
                         float* inv_q_fp32_scratch) {
    if (slab < 0 || slab >= bank->n_slabs) return;
    if (!q_vec || !v_out_fp32 || !scratch_4N || !int_scratch ||
        !inv_q_fp32_scratch) return;
    const int N = SP_ARM_RING_N;
    const int d = bank->d;

    uint64_t* InvQ_q1 = scratch_4N + 0 * N;
    uint64_t* InvQ_q2 = scratch_4N + 1 * N;
    uint64_t* Out_q1  = scratch_4N + 2 * N;
    uint64_t* Out_q2  = scratch_4N + 3 * N;

    /* Compute the negacyclic involution of q in fp32, then encode it
     * with a FORWARD encoder. We pass d=N to the encoder so it walks
     * the full length-N involution buffer (the involution is naturally
     * supported on positions [0] and [N-d+1..N-1]; positions in
     * [1..N-d] are zero). */
    sp_arm_involution_fp32(q_vec, d, N, inv_q_fp32_scratch);
    sp_poly_encode_ntt_q_crt(InvQ_q1, InvQ_q2,
                              inv_q_fp32_scratch, /*d=*/N,
                              bank->delta, int_scratch);

    const uint64_t* slab_q1 = bank->M_q1 + (size_t)slab * (size_t)N;
    const uint64_t* slab_q2 = bank->M_q2 + (size_t)slab * (size_t)N;

    /* Pointwise multiply M ⊙ Inv_q in each universe. */
    sp_ntt_crt_pointwise_mul(Out_q1, slab_q1, InvQ_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_pointwise_mul(Out_q2, slab_q2, InvQ_q2, &SP_NTT_CRT_CTX_Q2);

    /* Inverse NTT both back to coefficient domain. */
    sp_ntt_crt_inverse(Out_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_inverse(Out_q2, &SP_NTT_CRT_CTX_Q2);

    /* CRT stitch + decode: with the involution math, the recalled v_j
     * approximation lands at coefficients [0, d) directly, in their
     * natural order. Each is M(x)*inv(q)(x) mod x^N+1 evaluated at x^i,
     * approximately equal to ||q||^2 * v_j[i] + cross-talk noise.
     * Decode divides by delta^3 (K, V, inv(q) each contributed delta). */
    const double M_d = (double)SP_NTT_CRT_Q1 * (double)SP_NTT_CRT_Q2;
    const double inv_delta3 =
        1.0 / (bank->delta * bank->delta * bank->delta);
    for (int i = 0; i < d; ++i) {
        uint64_t u = sp_ntt_crt_combine(Out_q1[i], Out_q2[i]);
        double x;
        if ((double)u > 0.5 * M_d) {
            x = (double)u - M_d;
        } else {
            x = (double)u;
        }
        v_out_fp32[i] = (float)(x * inv_delta3);
    }
}

double sp_arm_bank_norm(const sp_arm_bank* bank, int slab,
                         uint64_t* scratch_2N) {
    if (slab < 0 || slab >= bank->n_slabs) return 0.0;
    if (!scratch_2N) return 0.0;
    const int N = SP_ARM_RING_N;
    uint64_t* C_q1 = scratch_2N + 0 * N;
    uint64_t* C_q2 = scratch_2N + 1 * N;

    /* Inverse NTT a COPY of the slab — we don't want to mutate the bank. */
    const uint64_t* slab_q1 = bank->M_q1 + (size_t)slab * (size_t)N;
    const uint64_t* slab_q2 = bank->M_q2 + (size_t)slab * (size_t)N;
    memcpy(C_q1, slab_q1, N * sizeof(uint64_t));
    memcpy(C_q2, slab_q2, N * sizeof(uint64_t));
    sp_ntt_crt_inverse(C_q1, &SP_NTT_CRT_CTX_Q1);
    sp_ntt_crt_inverse(C_q2, &SP_NTT_CRT_CTX_Q2);

    const double M_d = (double)SP_NTT_CRT_Q1 * (double)SP_NTT_CRT_Q2;
    const double inv_delta_sq = 1.0 / (bank->delta * bank->delta);
    double ss = 0.0;
    for (int i = 0; i < N; ++i) {
        uint64_t u = sp_ntt_crt_combine(C_q1[i], C_q2[i]);
        double x;
        if ((double)u > 0.5 * M_d) {
            x = (double)u - M_d;
        } else {
            x = (double)u;
        }
        /* Bank coefficient ~ k*v product → delta^2 scaling. */
        x *= inv_delta_sq;
        ss += x * x;
    }
    return ss;
}
