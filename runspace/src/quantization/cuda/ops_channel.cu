// runspace/src/quantization/cuda/ops_channel.cu
//
// Channel-mode codec kernels.

#include "codec.cuh"
#include "codec_launch.h"

#include <cuda_runtime.h>

namespace qbench_lp {

static constexpr int CHAN_BLK = 256;


static __global__ void encode_channel_kernel(
    const float*    __restrict__ x,
    std::uint32_t*  __restrict__ out,
    float*          __restrict__ scales,
    int C, int K_pad,
    int e, int m, int is_signed,
    int npw, int w)
{
    __shared__ float sdata[CHAN_BLK];
    __shared__ float inv_s_sh;

    const int c = blockIdx.x;
    if (c >= C) return;

    const float*    row  = x   + (size_t)c * (size_t)K_pad;
    const int       wpr  = K_pad / npw;
    std::uint32_t*  orow = out + (size_t)c * (size_t)wpr;

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K_pad; i += CHAN_BLK) {
        local_max = fmaxf(local_max, fabsf(row[i]));
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = CHAN_BLK >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float amax = sdata[0];
        if (amax == 0.0f) amax = 1.0f;
        std::uint32_t bits = __float_as_uint(amax);
        bits &= 0xFF800000u;
        const float scale = __uint_as_float(bits);
        scales[c] = scale;
        inv_s_sh  = 1.0f / scale;
    }
    __syncthreads();

    const float inv_s = inv_s_sh;

    for (int word = threadIdx.x; word < wpr; word += CHAN_BLK) {
        const int base = word * npw;
        std::uint32_t packed = 0u;
        for (int k = 0; k < npw; ++k) {
            const float v = row[base + k];
            const std::uint32_t f = encode_emb(v * inv_s, e, m, is_signed);
            packed |= (f << (k * w));
        }
        orow[word] = packed;
    }
}


__constant__ ChannelDecodeMeta g_meta;

static __global__ void decode_channel_kernel(
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scales,
    float*               __restrict__ out,
    int C, int K, int K_pad,
    int e, int m, int is_signed,
    int npw, int w)
{
    const int c = blockIdx.x;
    if (c >= C) return;

    const int wpr = K_pad / npw;
    const std::uint32_t* irow = in + (size_t)c * (size_t)wpr;
    const float          s    = scales[c];

    const int channel_dim = g_meta.channel_dim;
    const int n_other     = g_meta.n_other;
    const int c_stride    = g_meta.stride_orig[channel_dim];

    const std::uint32_t mask = (w == 32) ? 0xFFFFFFFFu : ((1u << w) - 1u);

    for (int word = threadIdx.x; word < wpr; word += CHAN_BLK) {
        const std::uint32_t pck = irow[word];
        const int           base = word * npw;
        for (int k = 0; k < npw; ++k) {
            const int kpos = base + k;
            if (kpos >= K) break;

            int orig_idx = c * c_stride;
            int rem      = kpos;
            for (int j = 0; j < n_other; ++j) {
                const int inner = g_meta.inner_stride[j];
                int coord;
                if (inner == 1) {
                    coord = rem;
                } else {
                    coord = rem / inner;
                    rem   = rem - coord * inner;
                }
                orig_idx += coord * g_meta.stride_orig[g_meta.other_axes[j]];
            }

            const std::uint32_t field = (pck >> (k * w)) & mask;
            out[orig_idx] = decode_emb(field, e, m, is_signed) * s;
        }
    }
}


void launch_encode_channel(
    const float*    x,
    std::uint32_t*  out,
    float*          scales,
    int             C, int K_pad,
    int             e, int m, int is_signed,
    void*           stream)
{
    if (C == 0 || K_pad == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int w   = element_width(e, m, is_signed);
    const int npw = n_per_word(w);
    encode_channel_kernel<<<C, CHAN_BLK, 0, cs>>>(
        x, out, scales, C, K_pad, e, m, is_signed, npw, w);
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
    if (C == 0 || K == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int w   = element_width(e, m, is_signed);
    const int npw = n_per_word(w);
    cudaMemcpyToSymbolAsync(g_meta, &meta, sizeof(meta), 0,
                            cudaMemcpyHostToDevice, cs);
    decode_channel_kernel<<<C, CHAN_BLK, 0, cs>>>(
        in, scales, out, C, K, K_pad, e, m, is_signed, npw, w);
}


}  // namespace qbench_lp
