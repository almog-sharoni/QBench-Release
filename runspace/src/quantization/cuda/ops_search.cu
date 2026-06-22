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

// Per-chunk error metrics used for format selection. These mirror
// DynamicInputQuantizer._METRIC_CODES on the Python side and must stay in sync.
// All are "lower is better" so the search keeps the per-chunk argmin.
enum SearchMetric {
    METRIC_L2     = 0,  // sum(diff^2)            -- reduce SUM (CUDA legacy default)
    METRIC_L1     = 1,  // sum(|diff|)            -- reduce SUM
    METRIC_LINF   = 2,  // max(|diff|)            -- reduce MAX
    METRIC_BIAS   = 3,  // |sum(diff)|            -- reduce SUM, then abs
    METRIC_L0     = 4,  // count(diff != 0)       -- reduce SUM
    METRIC_HUBER  = 5,  // sum(huber(diff,delta)) -- reduce SUM
    METRIC_LOGSUM = 6,  // sum(floor(log2|diff|)) -- reduce SUM (log-domain L1)
};

// Per-element contribution for the active metric (operates on the scaled error).
__device__ __forceinline__ float metric_elem(float diff, int metric, float param)
{
    switch (metric) {
        case METRIC_L1:   return fabsf(diff);
        case METRIC_LINF: return fabsf(diff);
        case METRIC_BIAS: return diff;                       // signed; abs applied to the sum
        case METRIC_L0:   return (diff != 0.0f) ? 1.0f : 0.0f;
        case METRIC_HUBER: {
            const float a = fabsf(diff);
            return (a <= param) ? (0.5f * diff * diff)
                                : (param * (a - 0.5f * param));
        }
        case METRIC_LOGSUM: {
            const float a = fabsf(diff);
            // Floor of the binary exponent; exact zeros get a finite floor so
            // formats that reproduce a value exactly are strongly preferred.
            return (a > 0.0f) ? floorf(log2f(a)) : -126.0f;
        }
        case METRIC_L2:
        default:          return diff * diff;
    }
}

// Block reduction across the 128 lanes. use_max selects max-reduce (L-inf);
// otherwise sum-reduce. All metric values are non-negative except BIAS, which
// uses sum-reduce so the signed cancellation is preserved.
__device__ __forceinline__ float block_reduce_metric(float val, bool use_max, int lane)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = use_max ? fmaxf(val, other) : (val + other);
    }
    __shared__ float smem[4];
    if (lane % 32 == 0) smem[lane / 32] = val;
    __syncthreads();

    float r = use_max ? -3e38f : 0.0f;
    if (lane < 4) r = smem[lane];
    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, r, offset);
        r = use_max ? fmaxf(r, other) : (r + other);
    }
    if (lane == 0) smem[0] = r;
    __syncthreads();
    r = smem[0];
    __syncthreads();
    return r;
}

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
    int             n_chunks,
    int             metric,
    float           metric_param)
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

    // Step 2: Loop over candidates to find the format minimizing the metric.
    const bool use_max = (metric == METRIC_LINF);
    float best_err = 3e38f; // infinity
    int best_c = 0;
    float best_qv = 0.0f;

    for (int c = 0; c < num_candidates; ++c) {
        int e = cands_e[c];
        int m = cands_m[c];
        int sgn = cands_sgn[c];

        // Quantize and dequantize
        std::uint32_t packed = encode_emb(scaled_v, e, m, sgn);
        float qv = decode_emb(packed, e, m, sgn);

        // Per-element metric contribution, then block reduce.
        const float diff = scaled_v - qv;
        float chunk_err = block_reduce_metric(
            metric_elem(diff, metric, metric_param), use_max, lane);
        if (metric == METRIC_BIAS) chunk_err = fabsf(chunk_err);

        if (chunk_err < best_err) {
            best_err = chunk_err;
            best_c = c;
            best_qv = qv;
        }
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
    int          metric,
    float        metric_param,
    void*        stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    // Launch 1 block per chunk, 128 threads per block.
    search_and_quantize_chunk_kernel<<<n_chunks, BLK, 0, cs>>>(
        x, cands_e, cands_m, cands_sgn, num_candidates,
        best_indices, best_scales, out, out_unscaled, N, n_chunks,
        metric, metric_param);
}

} // namespace qbench_lp
