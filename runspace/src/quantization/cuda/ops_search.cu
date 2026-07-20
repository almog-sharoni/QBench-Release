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
    METRIC_PSEUDO_MSE = 7,  // bit-level pseudo err2-err1 selector
    METRIC_PSEUDO_MSE2 = 8,  // weighted bit-level pseudo err2-err1 selector
    METRIC_PSEUDO_MSE3 = 9,  // exact sum(err2^2 - err1^2) selector
};

enum PseudoMse3FixedRounding {
    PSEUDO_MSE3_FIXED_FLOOR = 0,
    PSEUDO_MSE3_FIXED_NEAREST = 1,
};

enum PseudoMse3TieBreak {
    PSEUDO_MSE3_TIE_EXP1 = 0,
    PSEUDO_MSE3_TIE_EXP2 = 1,
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

__device__ __forceinline__ int pseudo_mse_mantissa_bit(uint32_t mant, int bit_index)
{
    if (bit_index < 1 || bit_index > 23) {
        return 0;
    }
    return static_cast<int>((mant >> (23 - bit_index)) & 1u);
}

__device__ __forceinline__ float pseudo_mse_mantissa_tail_value(
    uint32_t mant,
    int start_bit,
    int tail_bits)
{
    const int max_offset = (tail_bits < 0) ? 23 : tail_bits;
    float value = 0.0f;
    float weight = 1.0f;
    #pragma unroll
    for (int offset = 1; offset <= 23; ++offset) {
        if (offset > max_offset) {
            break;
        }
        value += static_cast<float>(pseudo_mse_mantissa_bit(mant, start_bit + offset)) * weight;
        weight *= 0.5f;
    }
    return value;
}

__device__ __forceinline__ float pseudo_mse_err2_minus_err1(
    float scaled_v,
    int m1,
    int m2,
    int sgn)
{
    (void)sgn;
    if (m2 != m1 - 1) {
        return 0.0f;
    }

    const uint32_t mag = __float_as_uint(scaled_v) & 0x7FFFFFFFu;
    const uint32_t exp_field = (mag >> 23) & 0xFFu;
    if (exp_field == 0u) {
        return 0.0f;
    }

    const int e_depth = 127 - static_cast<int>(exp_field);
    const uint32_t mant = mag & 0x7FFFFFu;

    if (e_depth == 0) {
        return static_cast<float>((mant >> (23 - m1)) & 1u);
    }
    if (e_depth == 1) {
        return 0.0f;
    }
    if (e_depth > 1 && e_depth < m1 + 1) {
        const int k = m1 + 1 - e_depth;
        return -static_cast<float>((mant >> (23 - k)) & 1u);
    }
    if (e_depth == m1 + 1) {
        return -1.0f;
    }
    return 0.0f;
}

__device__ __forceinline__ float pseudo_mse2_err2_minus_err1(
    float scaled_v,
    int m1,
    int m2,
    int sgn,
    int mantissa_window_bits)
{
    (void)sgn;
    if (m2 != m1 - 1) {
        return 0.0f;
    }
    if (mantissa_window_bits < 0) {
        return 0.0f;
    }
    const int tail_bits = (mantissa_window_bits == 0) ? -1 : (mantissa_window_bits - 1);
    const int hidden_tail_bits = (mantissa_window_bits == 0) ? -1 : mantissa_window_bits;

    const uint32_t mag = __float_as_uint(scaled_v) & 0x7FFFFFFFu;
    const uint32_t exp_field = (mag >> 23) & 0xFFu;
    if (exp_field == 0u) {
        return 0.0f;
    }

    const int e_depth = 127 - static_cast<int>(exp_field);
    const uint32_t mant = mag & 0x7FFFFFu;

    if (e_depth == 0) {
        const float x_m = static_cast<float>(pseudo_mse_mantissa_bit(mant, m1));
        return x_m * (x_m + pseudo_mse_mantissa_tail_value(mant, m1, tail_bits));
    }
    if (e_depth == 1) {
        return 0.0f;
    }
    if (e_depth > 1 && e_depth < m1 + 1) {
        const int k = m1 + 1 - e_depth;
        const float x_k = static_cast<float>(pseudo_mse_mantissa_bit(mant, k));
        return -(x_k * (x_k + pseudo_mse_mantissa_tail_value(mant, k, tail_bits)));
    }
    if (e_depth == m1 + 1) {
        return -(1.0f + pseudo_mse_mantissa_tail_value(mant, 0, hidden_tail_bits));
    }
    return 0.0f;
}

__device__ __forceinline__ float pseudo_mse3_err2_minus_err1(
    float scaled_v,
    int m1,
    int m2,
    int sgn)
{
    if (m2 != m1 - 1) {
        return 0.0f;
    }

    // pseudo_MSE3 selects the format from truncating candidate encodings.
    // The selected activation is encoded with round-to-nearest below when the
    // kernel writes best_qv, so truncation affects only the choosing mechanism.
    const std::uint32_t packed_e1 = encode_emb_trunc(scaled_v, 1, m1, sgn);
    const std::uint32_t packed_e2 = encode_emb_trunc(scaled_v, 2, m2, sgn);
    const float q_e1 = decode_emb(packed_e1, 1, m1, sgn);
    const float q_e2 = decode_emb(packed_e2, 2, m2, sgn);
    const float err1 = scaled_v - q_e1;
    const float err2 = scaled_v - q_e2;
    // Keep the two squares and subtraction separately rounded. With fast-math,
    // contraction can otherwise leave a tiny signed residual when q_e1 == q_e2;
    // fixed-point conversion can otherwise amplify that mathematical tie.
    const float err1_sq = __fmul_rn(err1, err1);
    const float err2_sq = __fmul_rn(err2, err2);
    const float diff = __fsub_rn(err2_sq, err1_sq);
    const float normalization = ldexpf(1.0f, 2 * m1);
    return __fmul_rn(diff, normalization);
}

__device__ __forceinline__ float pseudo_mse3_apply_bits_to_take(
    float diff,
    int bits_to_take,
    int fixed_rounding_mode)
{
    const float scale = ldexpf(1.0f, bits_to_take);
    const float scaled = __fmul_rn(diff, scale);
    int fixed;
    if (fixed_rounding_mode == PSEUDO_MSE3_FIXED_NEAREST) {
        // Match pseudo_mse3_fixed_point_from_diff: weight negative scaled
        // values before applying half-away-from-zero rounding.
        const float weighted_scaled =
            (scaled < 0.0f) ? __fmul_rn(scaled, 4.0f) : scaled;
        const float rounded_magnitude =
            floorf(__fadd_rn(fabsf(weighted_scaled), 0.5f));
        // Match the Python path's post-rounding weight for positive values.
        const float post_round_magnitude =
            (weighted_scaled > 0.0f)
                ? __fmul_rn(rounded_magnitude, 4.0f)
                : rounded_magnitude;
        const int fixed_magnitude = __float2int_rz(post_round_magnitude);
        fixed = (weighted_scaled < 0.0f) ? -fixed_magnitude : fixed_magnitude;
    } else {
        fixed = __float2int_rd(scaled);
    }
    return static_cast<float>(fixed);
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
    float           metric_param,
    int             pseudo_mse2_mantissa_window_bits,
    int             pseudo_mse3_fixed_rounding_mode,
    int             pseudo_mse3_tie_break_mode)
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
    if (metric == METRIC_PSEUDO_MSE ||
        metric == METRIC_PSEUDO_MSE2 ||
        metric == METRIC_PSEUDO_MSE3) {
        int e1_idx = -1;
        int e2_idx = -1;
        int exp1_m = -1;
        int exp2_m = -1;
        int pair_sgn = 1;

        // Find an exp=1 / exp=2 pair having:
        //   - the same sign configuration
        //   - exp2 mantissa width = exp1 mantissa width - 1
        for (int c1 = 0; c1 < num_candidates && e1_idx < 0; ++c1) {
            if (cands_e[c1] != 1) {
                continue;
            }

            const int m1 = cands_m[c1];
            const int sgn1 = cands_sgn[c1];

            for (int c2 = 0; c2 < num_candidates; ++c2) {
                if (cands_e[c2] == 2 &&
                    cands_sgn[c2] == sgn1 &&
                    cands_m[c2] == m1 - 1) {
                    e1_idx = c1;
                    e2_idx = c2;
                    exp1_m = m1;
                    exp2_m = cands_m[c2];
                    pair_sgn = sgn1;
                    break;
                }
            }
        }

        int best_c = 0;

        if (e1_idx >= 0 && e2_idx >= 0 && exp1_m >= 1 && exp2_m == exp1_m - 1) {
            float diff = 0.0f;
            if (metric == METRIC_PSEUDO_MSE3) {
                diff = pseudo_mse3_err2_minus_err1(
                    scaled_v,
                    exp1_m,
                    exp2_m,
                    pair_sgn);
                const int bits_to_take = static_cast<int>(metric_param + 0.5f);
                diff = pseudo_mse3_apply_bits_to_take(
                    diff,
                    bits_to_take,
                    pseudo_mse3_fixed_rounding_mode);
            } else if (metric == METRIC_PSEUDO_MSE2) {
                diff = pseudo_mse2_err2_minus_err1(
                    scaled_v,
                    exp1_m,
                    exp2_m,
                    pair_sgn,
                    pseudo_mse2_mantissa_window_bits);
            } else {
                diff = pseudo_mse_err2_minus_err1(
                    scaled_v,
                    exp1_m,
                    exp2_m,
                    pair_sgn);
            }

            if (metric == METRIC_PSEUDO_MSE3) {
                // pseudo_MSE3 is the exact signed squared-error difference.
                // Negative chunk sum means exp=2 has lower total squared error.
                const float chunk_diff = block_reduce_metric(diff, false, lane);
                const bool choose_e2 =
                    (chunk_diff < 0.0f) ||
                    (pseudo_mse3_tie_break_mode == PSEUDO_MSE3_TIE_EXP2 &&
                     chunk_diff == 0.0f);
                best_c = choose_e2 ? e2_idx : e1_idx;
            } else {
                // pseudo_MSE/pseudo_MSE2 are per-element winner votes, not summed MSE.
                // Sign convention:
                //   diff < 0 means exp=2 wins
                //   diff > 0 means exp=1 wins
                // exact ties do not vote. pseudo_MSE2 uses fractional weighted sums.
                const float exp1_vote = (diff > 0.0f) ? diff : 0.0f;
                const float exp2_vote = (diff < 0.0f) ? -diff : 0.0f;
                const float e1_wins = block_reduce_metric(exp1_vote, false, lane);
                const float e2_wins = block_reduce_metric(exp2_vote, false, lane);

                // Equivalent signed vote convention is exp=1 positive and
                // exp=2 negative. The decision uses explicit counts and divides
                // exp=2 wins by the pseudo_MSE divisor. A tie selects exp=2.
                int e2_win_divisor = static_cast<int>(metric_param + 0.5f);
                if (e2_win_divisor != 2 && e2_win_divisor != 4) {
                    e2_win_divisor = 4;
                }
                const float e2_wins_shifted = e2_wins / static_cast<float>(e2_win_divisor);
                best_c = (e2_wins_shifted >= e1_wins) ? e2_idx : e1_idx;
            }
        }

        const int e = cands_e[best_c];
        const int m = cands_m[best_c];
        const int sgn = cands_sgn[best_c];
        std::uint32_t packed = encode_emb(scaled_v, e, m, sgn);
        const float best_qv = decode_emb(packed, e, m, sgn);

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
        return;
    }

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
    int          pseudo_mse2_mantissa_window_bits,
    int          pseudo_mse3_fixed_rounding_mode,
    int          pseudo_mse3_tie_break_mode,
    void*        stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    // Launch 1 block per chunk, 128 threads per block.
    search_and_quantize_chunk_kernel<<<n_chunks, BLK, 0, cs>>>(
        x, cands_e, cands_m, cands_sgn, num_candidates,
        best_indices, best_scales, out, out_unscaled, N, n_chunks,
        metric, metric_param, pseudo_mse2_mantissa_window_bits,
        pseudo_mse3_fixed_rounding_mode, pseudo_mse3_tie_break_mode);
}

} // namespace qbench_lp
