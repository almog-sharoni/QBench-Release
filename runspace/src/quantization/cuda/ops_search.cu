// runspace/src/quantization/cuda/ops_search.cu
//
// Fused CUDA kernel for searching the optimal quantization format per chunk.

#include "codec.cuh"
#include "codec_launch.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace qbench_lp {

static constexpr int CHUNK            = 128;
static constexpr int WARP             = 32;
static constexpr int CHUNKS_PER_BLOCK = 1;
static constexpr int BLK              = WARP * 4;  // 128 threads per block

// This kernel computes the best format per chunk, and directly decodes the values.
// We use 128 threads per block. Each block processes exactly 1 chunk.
__global__ void search_and_quantize_chunk_kernel(
    const float*    __restrict__ x,
    const int*      __restrict__ cands_e,
    const int*      __restrict__ cands_m,
    const int*      __restrict__ cands_sgn,
    int             num_candidates,
    int64_t*        __restrict__ best_indices,
    float*          __restrict__ best_scales,
    float*          __restrict__ out,
    float*          __restrict__ out_unscaled, // can be nullptr
    int             N,
    int             n_chunks)
{
    const int chunk = blockIdx.x;
    if (chunk >= n_chunks) return;

    const int chunk_base = chunk * CHUNK;
    const int lane = threadIdx.x;
    const int global_idx = chunk_base + lane;

    // Load elements into registers
    float v = 0.0f;
    if (global_idx < N) {
        v = x[global_idx];
    }

    // Step 1: Compute amax and scale using warp reduce
    float local_max = fabsf(v);
    
    // Block reduction for amax
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    // Now lane 0, 32, 64, 96 have the warp maxes. Share them using shared memory.
    __shared__ float smem_max[4];
    if (lane % 32 == 0) smem_max[lane / 32] = local_max;
    __syncthreads();
    
    float amax = 0.0f;
    if (lane < 4) amax = smem_max[lane];
    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFF, amax, offset));
    }
    // Broadcast amax to all threads in block using shared memory
    if (lane == 0) smem_max[0] = amax;
    __syncthreads();
    amax = smem_max[0];

    const float s = pow2_floor_nonneg(amax);
    const float inv_s = 1.0f / s;
    const float scaled_v = v * inv_s;

    // Step 2: Loop over candidates to find best MSE
    float best_err = 3e38f; // infinity
    int best_c = -1;
    float best_qv = 0.0f;

    for (int c = 0; c < num_candidates; ++c) {
        int e = cands_e[c];
        int m = cands_m[c];
        int sgn = cands_sgn[c];

        // Quantize and dequantize
        std::uint32_t packed = encode_emb(scaled_v, e, m, sgn);
        float qv = decode_emb(packed, e, m, sgn);

        // Error calculation
        float diff = scaled_v - qv;
        float err = diff * diff;

        // Block reduction for sum(err)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            err += __shfl_down_sync(0xFFFFFFFF, err, offset);
        }
        __shared__ float smem_err[4];
        if (lane % 32 == 0) smem_err[lane / 32] = err;
        __syncthreads();

        float sum_err = 0.0f;
        if (lane < 4) sum_err = smem_err[lane];
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            sum_err += __shfl_down_sync(0xFFFFFFFF, sum_err, offset);
        }
        
        // Broadcast sum_err to all threads in block using shared memory
        if (lane == 0) smem_err[0] = sum_err;
        __syncthreads();
        sum_err = smem_err[0];

        if (sum_err < best_err) {
            best_err = sum_err;
            best_c = c;
            best_qv = qv;
        }
        
        // Ensure all threads have read smem_err[0] before the next iteration
        // overwrites smem_err in the warp reduction step.
        __syncthreads();
    }

    // Step 3: Write outputs
    if (global_idx < N) {
        out[global_idx] = best_qv * s;
        if (out_unscaled != nullptr) {
            out_unscaled[global_idx] = best_qv;
        }
    }
    if (lane == 0) {
        best_indices[chunk] = best_c;
        best_scales[chunk] = s;
    }
}

void launch_search_and_quantize_chunk(
    const float* x,
    const int*   cands_e,
    const int*   cands_m,
    const int*   cands_sgn,
    int          num_candidates,
    int64_t*     best_indices,
    float*       best_scales,
    float*       out,
    float*       out_unscaled,
    int          N,
    void*        stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    // Launch 1 block per chunk, 128 threads per block.
    search_and_quantize_chunk_kernel<<<n_chunks, BLK, 0, cs>>>(
        x, cands_e, cands_m, cands_sgn, num_candidates,
        best_indices, best_scales, out, out_unscaled, N, n_chunks);
}

} // namespace qbench_lp
