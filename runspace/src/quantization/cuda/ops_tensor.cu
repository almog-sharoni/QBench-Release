// runspace/src/quantization/cuda/ops_tensor.cu
//
// Tensor-mode codec kernels.  Encode/decode are per-format template
// specializations dispatched at the bottom of the file; the amax reduction
// is format-independent and stays as a regular kernel.

#include "codec.cuh"
#include "codec_launch.h"
#include "formats.def"

#include <cuda_runtime.h>

namespace qbench_lp {

static constexpr int AMAX_BLK  = 256;
static constexpr int FINAL_BLK = 256;
static constexpr int AMAX_VPT  = 4;
static constexpr int ENC_BLK   = 256;


// ----------------------------------------------------------------------------
// Format-independent amax reduction (warp-shuffle, two-pass).
// ----------------------------------------------------------------------------

static __global__ void reduce_amax_pass1(
    const float* __restrict__ x,
    float*       __restrict__ partial,
    int N)
{
    constexpr int N_WARPS = AMAX_BLK / 32;
    __shared__ float warp_max_sh[N_WARPS];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int block_base = blockIdx.x * AMAX_BLK * AMAX_VPT;
    const int gid        = block_base + tid * AMAX_VPT;

    float local_max = 0.0f;
    if (gid + AMAX_VPT <= N) {
        const float4 v = reinterpret_cast<const float4*>(x)[gid >> 2];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                          fmaxf(fabsf(v.z), fabsf(v.w)));
    } else {
        #pragma unroll
        for (int j = 0; j < AMAX_VPT; ++j) {
            const int idx = gid + j;
            if (idx < N) local_max = fmaxf(local_max, fabsf(x[idx]));
        }
    }

    const float warp_max = warp_amax(local_max);
    if (lane == 0) warp_max_sh[warp_id] = warp_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? warp_max_sh[lane] : 0.0f;
        const float block_max = warp_amax(v);
        if (lane == 0) partial[blockIdx.x] = block_max;
    }
}

static __global__ void reduce_amax_pass2(
    const float* __restrict__ partial,
    float*       __restrict__ scale_out,
    int n)
{
    constexpr int N_WARPS = FINAL_BLK / 32;
    __shared__ float warp_max_sh[N_WARPS];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    float my_max = 0.0f;
    for (int i = tid; i < n; i += FINAL_BLK)
        my_max = fmaxf(my_max, partial[i]);

    const float warp_max = warp_amax(my_max);
    if (lane == 0) warp_max_sh[warp_id] = warp_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? warp_max_sh[lane] : 0.0f;
        const float final_max = warp_amax(v);
        if (lane == 0) {
            float amax = final_max;
            if (amax == 0.0f) amax = 1.0f;
            std::uint32_t bits = __float_as_uint(amax);
            bits &= 0xFF800000u;
            scale_out[0] = __uint_as_float(bits);
        }
    }
}


// ----------------------------------------------------------------------------
// Encode / decode templates
// ----------------------------------------------------------------------------

template <int E, int M, int IS_SIGNED>
static __global__ void encode_tensor_kernel_t(
    const float*          __restrict__ x,
    const float*          __restrict__ scale_ptr,
    std::uint32_t*        __restrict__ out,
    int N)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = (N + NPW - 1) / NPW;
    if (gid >= total_words) return;

    const float inv_s = 1.0f / scale_ptr[0];
    const int   base  = gid * NPW;

    std::uint32_t packed = 0u;
    #pragma unroll
    for (int lane = 0; lane < NPW; ++lane) {
        const int idx = base + lane;
        const float v = (idx < N) ? x[idx] : 0.0f;
        const std::uint32_t f = encode_emb(v * inv_s, E, M, IS_SIGNED);
        packed |= (f << (lane * W));
    }
    out[gid] = packed;
}

template <int E, int M, int IS_SIGNED>
static __global__ void decode_tensor_kernel_t(
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scale_ptr,
    float*               __restrict__ out,
    int N)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    constexpr std::uint32_t MASK = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = (N + NPW - 1) / NPW;
    if (gid >= total_words) return;

    const float          s    = scale_ptr[0];
    const std::uint32_t  pck  = in[gid];
    const int            base = gid * NPW;

    #pragma unroll
    for (int lane = 0; lane < NPW; ++lane) {
        const int idx = base + lane;
        if (idx >= N) break;
        const std::uint32_t field = (pck >> (lane * W)) & MASK;
        out[idx] = decode_emb(field, E, M, IS_SIGNED) * s;
    }
}


template <int E, int M, int IS_SIGNED>
static void launch_encode_tensor_t(
    const float* x, const float* scale_ptr, std::uint32_t* out,
    int N, void* stream)
{
    if (N == 0) return;
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    auto cs = static_cast<cudaStream_t>(stream);
    const int total_words = (N + NPW - 1) / NPW;
    const int grid = (total_words + ENC_BLK - 1) / ENC_BLK;
    encode_tensor_kernel_t<E, M, IS_SIGNED><<<grid, ENC_BLK, 0, cs>>>(
        x, scale_ptr, out, N);
}

template <int E, int M, int IS_SIGNED>
static void launch_decode_tensor_t(
    const std::uint32_t* in, const float* scale_ptr, float* out,
    int N, void* stream)
{
    if (N == 0) return;
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    auto cs = static_cast<cudaStream_t>(stream);
    const int total_words = (N + NPW - 1) / NPW;
    const int grid = (total_words + ENC_BLK - 1) / ENC_BLK;
    decode_tensor_kernel_t<E, M, IS_SIGNED><<<grid, ENC_BLK, 0, cs>>>(
        in, scale_ptr, out, N);
}


// ----------------------------------------------------------------------------
// Public launchers
// ----------------------------------------------------------------------------

void launch_compute_scale_tensor(
    const float* x,
    float*       partial,
    float*       scale_out,
    int          N,
    void*        stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int per_block = AMAX_BLK * AMAX_VPT;
    const int grid1 = (N + per_block - 1) / per_block;
    reduce_amax_pass1<<<grid1, AMAX_BLK, 0, cs>>>(x, partial, N);
    reduce_amax_pass2<<<1, FINAL_BLK, 0, cs>>>(partial, scale_out, grid1);
}

#define LP_KEY(E, M, S) (((E) << 6) | ((M) << 1) | ((S) & 1))

void launch_encode_tensor(
    const float*    x,
    const float*    scale_ptr,
    std::uint32_t*  out,
    int             N,
    int             e, int m, int is_signed,
    void*           stream)
{
    const int key = LP_KEY(e, m, is_signed);
    switch (key) {
    #define CASE(E, M, S) case LP_KEY(E, M, S): \
        launch_encode_tensor_t<E, M, S>(x, scale_ptr, out, N, stream); return;
    LP_FORMAT_LIST(CASE)
    #undef CASE
    }
}

void launch_decode_tensor(
    const std::uint32_t* in,
    const float*         scale_ptr,
    float*               out,
    int                  N,
    int                  e, int m, int is_signed,
    void*                stream)
{
    const int key = LP_KEY(e, m, is_signed);
    switch (key) {
    #define CASE(E, M, S) case LP_KEY(E, M, S): \
        launch_decode_tensor_t<E, M, S>(in, scale_ptr, out, N, stream); return;
    LP_FORMAT_LIST(CASE)
    #undef CASE
    }
}


}  // namespace qbench_lp
