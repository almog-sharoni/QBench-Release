// runspace/src/quantization/cuda/ops_channel.cu
//
// Channel-mode codec kernels.  Per-format template specialization; the
// dispatcher at the bottom routes (e, m, is_signed) to the matching
// encode/decode template instantiation.

#include "codec.cuh"
#include "codec_launch.h"
#include "formats.def"

#include <cuda_runtime.h>

namespace qbench_lp {

static constexpr int CHAN_BLK = 256;
static constexpr int CHAN_NW  = CHAN_BLK / 32;


template <int E, int M, int IS_SIGNED>
static __global__ void encode_channel_kernel_t(
    const float*    __restrict__ x,
    std::uint32_t*  __restrict__ out,
    float*          __restrict__ scales,
    int C, int K_pad)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;

    __shared__ float warp_max_sh[CHAN_NW];
    __shared__ float inv_s_sh;

    const int c = blockIdx.x;
    if (c >= C) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const float*    row  = x   + (size_t)c * (size_t)K_pad;
    const int       wpr  = K_pad / NPW;
    std::uint32_t*  orow = out + (size_t)c * (size_t)wpr;

    float local_max = 0.0f;
    const int K4 = K_pad >> 2;
    if ((K_pad & 3) == 0 && K4 >= CHAN_BLK) {
        const float4* rowv = reinterpret_cast<const float4*>(row);
        for (int i = tid; i < K4; i += CHAN_BLK) {
            const float4 v = rowv[i];
            const float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                                  fmaxf(fabsf(v.z), fabsf(v.w)));
            local_max = fmaxf(local_max, a);
        }
    } else {
        for (int i = tid; i < K_pad; i += CHAN_BLK) {
            local_max = fmaxf(local_max, fabsf(row[i]));
        }
    }

    const float warp_max = warp_amax(local_max);
    if (lane == 0) warp_max_sh[warp_id] = warp_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < CHAN_NW) ? warp_max_sh[lane] : 0.0f;
        const float row_max = warp_amax(v);
        if (lane == 0) {
            float amax = row_max;
            if (amax == 0.0f) amax = 1.0f;
            unsigned bits = __float_as_uint(amax);
            bits &= 0xFF800000u;
            const float scale = __uint_as_float(bits);
            scales[c] = scale;
            inv_s_sh  = 1.0f / scale;
        }
    }
    __syncthreads();

    const float inv_s = inv_s_sh;

    for (int word = tid; word < wpr; word += CHAN_BLK) {
        const int base = word * NPW;
        std::uint32_t packed = 0u;
        #pragma unroll
        for (int k = 0; k < NPW; ++k) {
            const float v = row[base + k];
            const std::uint32_t f = encode_emb(v * inv_s, E, M, IS_SIGNED);
            packed |= (f << (k * W));
        }
        orow[word] = packed;
    }
}


// 2-D fast path: tile the (k, c) output grid so that the warp's unit-stride
// dim is the writes' inner axis (c_stride == 1 → vary c in warp;
// c_stride > 1 → vary k in warp).  This restores coalesced output writes.
// Used when the host detects n_other == 1.
//
// TILE_C × TILE_K = 32 × 8 = 256 threads/block.

static constexpr int TILE_C = 32;
static constexpr int TILE_K = 8;

template <int E, int M, int IS_SIGNED>
static __global__ void decode_channel_2d_innerC_kernel_t(
    // c_stride == 1, other_stride == C, output layout = (K, C)
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scales,
    float*               __restrict__ out,
    int C, int K, int wpr)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    constexpr std::uint32_t MASK = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);

    const int c = blockIdx.x * TILE_C + threadIdx.x;
    const int k = blockIdx.y * TILE_K + threadIdx.y;
    if (c >= C || k >= K) return;

    const int word_in_row = k / NPW;
    const int kmod        = k - word_in_row * NPW;
    const std::uint32_t pck   = in[(size_t)c * (size_t)wpr + word_in_row];
    const std::uint32_t field = (pck >> (kmod * W)) & MASK;
    out[(size_t)k * (size_t)C + c] =
        decode_emb(field, E, M, IS_SIGNED) * scales[c];
}

template <int E, int M, int IS_SIGNED>
static __global__ void decode_channel_2d_outerC_kernel_t(
    // c_stride == K, other_stride == 1, output layout = (C, K)
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scales,
    float*               __restrict__ out,
    int C, int K, int wpr)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    constexpr std::uint32_t MASK = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);

    const int k = blockIdx.x * TILE_C + threadIdx.x;
    const int c = blockIdx.y * TILE_K + threadIdx.y;
    if (c >= C || k >= K) return;

    const int word_in_row = k / NPW;
    const int kmod        = k - word_in_row * NPW;
    const std::uint32_t pck   = in[(size_t)c * (size_t)wpr + word_in_row];
    const std::uint32_t field = (pck >> (kmod * W)) & MASK;
    out[(size_t)c * (size_t)K + k] =
        decode_emb(field, E, M, IS_SIGNED) * scales[c];
}


template <int E, int M, int IS_SIGNED>
static __global__ void decode_channel_kernel_t(
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scales,
    float*               __restrict__ out,
    int C, int K, int K_pad,
    ChannelDecodeMeta meta)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    constexpr std::uint32_t MASK = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);

    const int c = blockIdx.x;
    if (c >= C) return;

    const int wpr = K_pad / NPW;
    const std::uint32_t* irow = in + (size_t)c * (size_t)wpr;
    const float          s    = scales[c];

    const int channel_dim = meta.channel_dim;
    const int n_other     = meta.n_other;
    const int c_stride    = meta.stride_orig[channel_dim];

    for (int word = threadIdx.x; word < wpr; word += CHAN_BLK) {
        const std::uint32_t pck = irow[word];
        const int           base = word * NPW;
        #pragma unroll
        for (int k = 0; k < NPW; ++k) {
            const int kpos = base + k;
            if (kpos >= K) break;

            int orig_idx = c * c_stride;
            int rem      = kpos;
            for (int j = 0; j < n_other; ++j) {
                const int inner = meta.inner_stride[j];
                int coord;
                if (inner == 1) {
                    coord = rem;
                } else {
                    coord = rem / inner;
                    rem   = rem - coord * inner;
                }
                orig_idx += coord * meta.stride_orig[meta.other_axes[j]];
            }

            const std::uint32_t field = (pck >> (k * W)) & MASK;
            out[orig_idx] = decode_emb(field, E, M, IS_SIGNED) * s;
        }
    }
}


template <int E, int M, int IS_SIGNED>
static void launch_encode_channel_t(
    const float* x, std::uint32_t* out, float* scales,
    int C, int K_pad, void* stream)
{
    if (C == 0 || K_pad == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    encode_channel_kernel_t<E, M, IS_SIGNED><<<C, CHAN_BLK, 0, cs>>>(
        x, out, scales, C, K_pad);
}

template <int E, int M, int IS_SIGNED>
static void launch_decode_channel_t(
    const std::uint32_t* in, const float* scales, float* out,
    int C, int K, int K_pad,
    const ChannelDecodeMeta& meta, void* stream)
{
    if (C == 0 || K == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    if (meta.n_other == 1) {
        constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
        constexpr int NPW = 32 / W;
        const int wpr = K_pad / NPW;
        const int c_stride = meta.stride_orig[meta.channel_dim];
        if (c_stride == 1) {
            // Channel is the unit-stride axis: warp varies c, writes coalesced.
            dim3 block(TILE_C, TILE_K);
            dim3 grid((C + TILE_C - 1) / TILE_C, (K + TILE_K - 1) / TILE_K);
            decode_channel_2d_innerC_kernel_t<E, M, IS_SIGNED>
                <<<grid, block, 0, cs>>>(in, scales, out, C, K, wpr);
        } else {
            // Other axis is unit-stride: warp varies k, writes coalesced.
            dim3 block(TILE_C, TILE_K);
            dim3 grid((K + TILE_C - 1) / TILE_C, (C + TILE_K - 1) / TILE_K);
            decode_channel_2d_outerC_kernel_t<E, M, IS_SIGNED>
                <<<grid, block, 0, cs>>>(in, scales, out, C, K, wpr);
        }
        return;
    }
    decode_channel_kernel_t<E, M, IS_SIGNED><<<C, CHAN_BLK, 0, cs>>>(
        in, scales, out, C, K, K_pad, meta);
}


// ----------------------------------------------------------------------------
// Public dispatchers
// ----------------------------------------------------------------------------

#define LP_KEY(E, M, S) (((E) << 6) | ((M) << 1) | ((S) & 1))

void launch_encode_channel(
    const float*    x,
    std::uint32_t*  out,
    float*          scales,
    int             C, int K_pad,
    int             e, int m, int is_signed,
    void*           stream)
{
    const int key = LP_KEY(e, m, is_signed);
    switch (key) {
    #define CASE(E, M, S) case LP_KEY(E, M, S): \
        launch_encode_channel_t<E, M, S>(x, out, scales, C, K_pad, stream); return;
    LP_FORMAT_LIST(CASE)
    #undef CASE
    }
}

void launch_decode_channel(
    const std::uint32_t*       in,
    const float*               scales,
    float*                     out,
    int                        C, int K, int K_pad,
    int                        e, int m, int is_signed,
    const ChannelDecodeMeta&   meta,
    void*                      stream)
{
    const int key = LP_KEY(e, m, is_signed);
    switch (key) {
    #define CASE(E, M, S) case LP_KEY(E, M, S): \
        launch_decode_channel_t<E, M, S>(in, scales, out, C, K, K_pad, meta, stream); return;
    LP_FORMAT_LIST(CASE)
    #undef CASE
    }
}


}  // namespace qbench_lp
