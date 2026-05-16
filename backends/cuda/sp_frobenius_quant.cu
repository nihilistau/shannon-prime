// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// sp_frobenius_quant.cu — CUDA kernel implementations.
//
// Strategy: O_K arithmetic on device is pure integer (int64 multiply-add),
// so the kernel is bandwidth-bound. One thread per state element; no shared
// memory needed (each thread reads its element, multiplies by a precomputed
// scalar or pi^k, writes back).
//
// Precomputation of pi^k and the inert scalar happens on host (cheap) and
// is passed as a kernel argument. The kernel itself never branches on prime
// type — it does either "scalar multiply" or "ring multiply" based on which
// launcher invoked it.

#include "sp_frobenius_quant.h"
#include "../../core/sp_frobenius.h"

#include <cuda_runtime.h>

// --- Device helpers --------------------------------------------------------

// Same multiplication as host: (a1 + b1 w)(a2 + b2 w) with w^2 = w - 41.
__device__ __forceinline__ sp_ok_t d_ok_mul(sp_ok_t x, sp_ok_t y) {
    sp_ok_t r;
    r.a = x.a * y.a - SP_OK_OMEGA_NORM * x.b * y.b;
    r.b = x.a * y.b + y.a * x.b + x.b * y.b;
    return r;
}

__device__ __forceinline__ sp_ok_t d_ok_scalar_mul(sp_ok_t x, int64_t s) {
    sp_ok_t r = { x.a * s, x.b * s };
    return r;
}

// --- Kernels ---------------------------------------------------------------

// Split-prime channel: state[i] = state[i] * pi_pow.
__global__ void k_frobenius_split(sp_ok_t * __restrict__ state,
                                   size_t n_elements, sp_ok_t pi_pow) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    state[idx] = d_ok_mul(state[idx], pi_pow);
}

// Inert-prime channel: state[i] = state[i] * scalar (where scalar = (-p)^(k/2)).
__global__ void k_frobenius_inert(sp_ok_t * __restrict__ state,
                                   size_t n_elements, int64_t scalar) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    state[idx] = d_ok_scalar_mul(state[idx], scalar);
}

// --- Launchers -------------------------------------------------------------

static int launch_split(sp_ok_t *d_state, size_t n_elements,
                         int64_t p, int64_t k, cudaStream_t stream) {
    sp_ok_t pi;
    if (!sp_find_element_of_norm(p, &pi)) return -2;
    sp_ok_t pi_pow = sp_ok_pow(pi, k);

    const int THREADS = 256;
    size_t blocks = (n_elements + THREADS - 1) / THREADS;
    k_frobenius_split<<<(unsigned int)blocks, THREADS, 0, stream>>>(
        d_state, n_elements, pi_pow);
    return 0;
}

static int launch_inert(sp_ok_t *d_state, size_t n_elements,
                         int64_t p, int64_t k, cudaStream_t stream) {
    if (k & 1) return -3;  // odd power not in O_K
    int64_t m = k / 2;
    int64_t scalar = 1;
    int64_t base = -p;
    while (m > 0) {
        if (m & 1) scalar *= base;
        base *= base;
        m >>= 1;
    }
    const int THREADS = 256;
    size_t blocks = (n_elements + THREADS - 1) / THREADS;
    k_frobenius_inert<<<(unsigned int)blocks, THREADS, 0, stream>>>(
        d_state, n_elements, scalar);
    return 0;
}

extern "C" int sp_cuda_frobenius_quant(sp_ok_t *d_state, size_t n_elements,
                                        int64_t p, int64_t k,
                                        void *stream) {
    if (d_state == NULL || n_elements == 0 || k < 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    if (sp_is_split(p))  return launch_split(d_state, n_elements, p, k, s);
    if (sp_is_inert(p))  return launch_inert(d_state, n_elements, p, k, s);
    return -4;  // ramified (or invalid)
}

extern "C" int sp_cuda_sato_tate_mix(sp_ok_t *d_state, size_t n_elements,
                                      int64_t p1, int64_t k1,
                                      int64_t p2, int64_t k2,
                                      void *stream) {
    int e1 = sp_cuda_frobenius_quant(d_state, n_elements, p1, k1, stream);
    if (e1 != 0) return e1;
    int e2 = sp_cuda_frobenius_quant(d_state, n_elements, p2, k2, stream);
    return e2;
}
