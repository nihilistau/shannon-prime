// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// sp_frobenius_quant.h — CUDA kernels for Paper D's --frobenius-quant
// and --sato-tate-mix flags. Bit-exact mirror of sp_frobenius.c.

#ifndef SHANNON_PRIME_FROBENIUS_QUANT_CUDA_H
#define SHANNON_PRIME_FROBENIUS_QUANT_CUDA_H

#include "../../core/sp_ok_arith.h"

#ifdef __cplusplus
extern "C" {
#endif

// Launch the CUDA kernel that applies phi_p^k to every state element of a
// tensor on the GPU.
//
// d_state: device pointer to n_elements of sp_ok_t (16 bytes each on 64-bit).
// p, k:    Frobenius parameters. p must be split (k can be arbitrary) or
//          inert with even k.
// stream:  CUDA stream (NULL = default).
//
// Internally precomputes pi^k or scalar (-p)^(k/2) on host, then launches.
// Returns 0 on success, non-zero on parameter error.
int sp_cuda_frobenius_quant(sp_ok_t *d_state, size_t n_elements,
                            int64_t p, int64_t k,
                            void *stream);

// Composite phi_p1^k1 ∘ phi_p2^k2 (Config E).
int sp_cuda_sato_tate_mix(sp_ok_t *d_state, size_t n_elements,
                          int64_t p1, int64_t k1,
                          int64_t p2, int64_t k2,
                          void *stream);

#ifdef __cplusplus
}
#endif

#endif  // SHANNON_PRIME_FROBENIUS_QUANT_CUDA_H
