// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CRT Multi-GPU — CUDA Modular MatMul Kernels
//
// Two kernel variants:
//   1. Mersenne (M1 = 2^31-1): bit-shift reduction in the inner loop
//   2. Generic modular (M2 = 2^31-19): uint64 accumulation + final %
//
// Both use tiled shared-memory strategy (TILE_DIM × TILE_DIM) for
// memory coalescing. The uint32 datatype means we get 2× the elements
// per cacheline vs fp32 (same bit width, but the reduction overhead
// is compensated by the CRT parallelism across two GPUs).

#include "sp_crt.h"
#include <cuda_runtime.h>
#include <stdint.h>

#define CRT_TILE_DIM 16

// ============================================================================
// Device-side Mersenne reduction
// ============================================================================

__device__ __forceinline__ uint32_t d_mersenne_reduce(uint64_t x) {
    uint32_t lo = (uint32_t)(x & 0x7FFFFFFFU);
    uint32_t hi = (uint32_t)(x >> 31);
    uint32_t r = lo + hi;
    return r >= SP_CRT_M1 ? r - (uint32_t)SP_CRT_M1 : r;
}

// ============================================================================
// Kernel: Tiled Mersenne matmul — C = (A × B) mod (2^31 - 1)
// ============================================================================
//
// A: [M × K] uint32, B: [K × N] uint32, C: [M × N] uint32
// Uses shared memory tiles of TILE_DIM × TILE_DIM.
// Mersenne reduction every TILE_DIM accumulations to prevent overflow.

__global__ void kernel_matmul_mersenne(const uint32_t * __restrict__ A,
                                        const uint32_t * __restrict__ B,
                                        uint32_t * __restrict__ C,
                                        int M, int N, int K) {
    __shared__ uint32_t sA[CRT_TILE_DIM][CRT_TILE_DIM];
    __shared__ uint32_t sB[CRT_TILE_DIM][CRT_TILE_DIM];

    int row = blockIdx.y * CRT_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * CRT_TILE_DIM + threadIdx.x;

    uint64_t acc = 0;
    int n_tiles = (K + CRT_TILE_DIM - 1) / CRT_TILE_DIM;

    for (int t = 0; t < n_tiles; t++) {
        int ak = t * CRT_TILE_DIM + threadIdx.x;
        int bk = t * CRT_TILE_DIM + threadIdx.y;

        // Load tile from A
        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? A[row * K + ak] : 0;
        // Load tile from B
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? B[bk * N + col] : 0;

        __syncthreads();

        // Accumulate dot product for this tile
        #pragma unroll
        for (int k = 0; k < CRT_TILE_DIM; k++) {
            acc += (uint64_t)sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Mersenne reduce after each tile to keep acc < 2^62
        acc = d_mersenne_reduce(acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = d_mersenne_reduce(acc);
    }
}

// ============================================================================
// Kernel: Tiled generic modular matmul — C = (A × B) mod m
// ============================================================================
//
// Same tiled structure but uses standard % for non-Mersenne modulus.

__global__ void kernel_matmul_mod(const uint32_t * __restrict__ A,
                                   const uint32_t * __restrict__ B,
                                   uint32_t * __restrict__ C,
                                   int M, int N, int K,
                                   uint64_t modulus) {
    __shared__ uint32_t sA[CRT_TILE_DIM][CRT_TILE_DIM];
    __shared__ uint32_t sB[CRT_TILE_DIM][CRT_TILE_DIM];

    int row = blockIdx.y * CRT_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * CRT_TILE_DIM + threadIdx.x;

    uint64_t acc = 0;
    int n_tiles = (K + CRT_TILE_DIM - 1) / CRT_TILE_DIM;

    for (int t = 0; t < n_tiles; t++) {
        int ak = t * CRT_TILE_DIM + threadIdx.x;
        int bk = t * CRT_TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? A[row * K + ak] : 0;
        sB[threadIdx.y][threadIdx.x] =
            (bk < K && col < N) ? B[bk * N + col] : 0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < CRT_TILE_DIM; k++) {
            acc += (uint64_t)sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Reduce after each tile
        acc %= modulus;

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = (uint32_t)(acc % modulus);
    }
}

// ============================================================================
// Kernel: Float-to-residue quantization
// ============================================================================

__global__ void kernel_quantize(const float * __restrict__ input,
                                 uint32_t * __restrict__ output,
                                 int n,
                                 double scale, int64_t zero_point,
                                 uint64_t modulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t ival = (int64_t)((double)input[idx] * scale + 0.5) + zero_point;
    if (ival < 0) ival = 0;
    output[idx] = (uint32_t)((uint64_t)ival % modulus);
}

// ============================================================================
// Host launchers
// ============================================================================

extern "C" void sp_crt_cuda_matmul_mersenne(const uint32_t *d_A,
                                             const uint32_t *d_B,
                                             uint32_t *d_C,
                                             int M, int N, int K,
                                             void *stream) {
    dim3 block(CRT_TILE_DIM, CRT_TILE_DIM);
    dim3 grid((N + CRT_TILE_DIM - 1) / CRT_TILE_DIM,
              (M + CRT_TILE_DIM - 1) / CRT_TILE_DIM);

    kernel_matmul_mersenne<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_A, d_B, d_C, M, N, K);
}

extern "C" void sp_crt_cuda_matmul_mod(const uint32_t *d_A,
                                        const uint32_t *d_B,
                                        uint32_t *d_C,
                                        int M, int N, int K,
                                        uint64_t modulus,
                                        void *stream) {
    dim3 block(CRT_TILE_DIM, CRT_TILE_DIM);
    dim3 grid((N + CRT_TILE_DIM - 1) / CRT_TILE_DIM,
              (M + CRT_TILE_DIM - 1) / CRT_TILE_DIM);

    kernel_matmul_mod<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_A, d_B, d_C, M, N, K, modulus);
}

extern "C" void sp_crt_cuda_quantize(const float *d_input,
                                      uint32_t *d_output,
                                      int n,
                                      double scale, int64_t zero_point,
                                      uint64_t modulus,
                                      void *stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    kernel_quantize<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        d_input, d_output, n, scale, zero_point, modulus);
}
