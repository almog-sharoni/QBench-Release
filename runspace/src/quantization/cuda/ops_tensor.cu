// runspace/src/quantization/cuda/ops_tensor.cu
//
// Tensor-mode codec kernels.  Each thread handles one packed uint32
// containing n_per_word(w) consecutive elements; w = signed_bit + e + m.

#include "codec.cuh"
#include "codec_launch.h"

#include <cuda_runtime.h>

namespace qbench_lp {

static constexpr int AMAX_BLK  = 256;
static constexpr int FINAL_BLK = 1024;


// ----------------------------------------------------------------------------
// Two-pass amax reduction (independent of the encoded layout)
// ----------------------------------------------------------------------------

static __global__ void reduce_amax_pass1(
    const float* __restrict__ x,
    float*       __restrict__ partial,
    int N)
{
    __shared__ float sdata[AMAX_BLK];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * AMAX_BLK + tid;
    sdata[tid] = (gid < N) ? fabsf(x[gid]) : 0.0f;
    __syncthreads();
    for (int s = AMAX_BLK >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

static __global__ void reduce_amax_pass2(
    const float* __restrict__ partial,
    float*       __restrict__ scale_out,
    int n)
{
    __shared__ float sdata[FINAL_BLK];
    const int tid = threadIdx.x;

    float my_max = 0.0f;
    for (int i = tid; i < n; i += FINAL_BLK)
        my_max = fmaxf(my_max, partial[i]);
    sdata[tid] = my_max;
    __syncthreads();

    for (int s = FINAL_BLK >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) {
        float amax = sdata[0];
        if (amax == 0.0f) amax = 1.0f;
        std::uint32_t bits = __float_as_uint(amax);
        bits &= 0xFF800000u;
        scale_out[0] = __uint_as_float(bits);
    }
}


// ----------------------------------------------------------------------------
// Encode / decode kernels
// ----------------------------------------------------------------------------

static __global__ void encode_tensor_kernel(
    const float*          __restrict__ x,
    const float*          __restrict__ scale_ptr,
    std::uint32_t*        __restrict__ out,
    int N,
    int e, int m, int is_signed,
    int npw,
    int w)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = (N + npw - 1) / npw;
    if (gid >= total_words) return;

    const float inv_s = 1.0f / scale_ptr[0];
    const int   base  = gid * npw;

    std::uint32_t packed = 0u;
    for (int lane = 0; lane < npw; ++lane) {
        const int idx = base + lane;
        const float v = (idx < N) ? x[idx] : 0.0f;
        const std::uint32_t f = encode_emb(v * inv_s, e, m, is_signed);
        packed |= (f << (lane * w));
    }
    out[gid] = packed;
}

static __global__ void decode_tensor_kernel(
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scale_ptr,
    float*               __restrict__ out,
    int N,
    int e, int m, int is_signed,
    int npw,
    int w)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_words = (N + npw - 1) / npw;
    if (gid >= total_words) return;

    const float          s    = scale_ptr[0];
    const std::uint32_t  pck  = in[gid];
    const int            base = gid * npw;
    const std::uint32_t  mask = (w == 32) ? 0xFFFFFFFFu : ((1u << w) - 1u);

    for (int lane = 0; lane < npw; ++lane) {
        const int idx = base + lane;
        if (idx >= N) break;
        const std::uint32_t field = (pck >> (lane * w)) & mask;
        out[idx] = decode_emb(field, e, m, is_signed) * s;
    }
}


// ----------------------------------------------------------------------------
// Launchers
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
    const int grid1 = (N + AMAX_BLK - 1) / AMAX_BLK;
    reduce_amax_pass1<<<grid1, AMAX_BLK, 0, cs>>>(x, partial, N);
    reduce_amax_pass2<<<1,     FINAL_BLK, 0, cs>>>(partial, scale_out, grid1);
}

void launch_encode_tensor(
    const float*    x,
    const float*    scale_ptr,
    std::uint32_t*  out,
    int             N,
    int             e, int m, int is_signed,
    void*           stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int w   = element_width(e, m, is_signed);
    const int npw = n_per_word(w);
    const int total_words = (N + npw - 1) / npw;
    const int grid = (total_words + AMAX_BLK - 1) / AMAX_BLK;
    encode_tensor_kernel<<<grid, AMAX_BLK, 0, cs>>>(
        x, scale_ptr, out, N, e, m, is_signed, npw, w);
}

void launch_decode_tensor(
    const std::uint32_t* in,
    const float*         scale_ptr,
    float*               out,
    int                  N,
    int                  e, int m, int is_signed,
    void*                stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int w   = element_width(e, m, is_signed);
    const int npw = n_per_word(w);
    const int total_words = (N + npw - 1) / npw;
    const int grid = (total_words + AMAX_BLK - 1) / AMAX_BLK;
    decode_tensor_kernel<<<grid, AMAX_BLK, 0, cs>>>(
        in, scale_ptr, out, N, e, m, is_signed, npw, w);
}


}  // namespace qbench_lp
