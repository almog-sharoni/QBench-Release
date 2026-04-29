// runspace/src/quantization/cuda/ops_chunk.cu
//
// Chunk-mode codec kernels (chunk_size = 128).

#include "codec.cuh"
#include "codec_launch.h"

#include <cuda_runtime.h>

namespace qbench_lp {

static constexpr int CHUNK = 128;
static constexpr int BLK   = 32;


static __global__ void encode_chunk_kernel(
    const float*    __restrict__ x,
    std::uint32_t*  __restrict__ out,
    float*          __restrict__ scales,
    int N,
    int e, int m, int is_signed,
    int npw, int wpc, int w)
{
    const int chunk = blockIdx.x;
    const int lane  = threadIdx.x;
    const int chunk_base = chunk * CHUNK;

    // Phase 1: per-chunk amax.  Each lane covers 4 elements (32 * 4 = 128).
    float local_max = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = chunk_base + lane + j * BLK;
        const float v = (idx < N) ? x[idx] : 0.0f;
        local_max = fmaxf(local_max, fabsf(v));
    }
    const float amax  = warp_amax(local_max);
    const float s     = pow2_floor_nonneg(amax);
    const float inv_s = 1.0f / s;

    if (lane == 0) scales[chunk] = s;

    std::uint32_t* o_chunk = out + (size_t)chunk * (size_t)wpc;

    for (int word = lane; word < wpc; word += BLK) {
        const int elem_base = word * npw;
        std::uint32_t packed = 0u;
        for (int k = 0; k < npw; ++k) {
            const int local_idx = elem_base + k;
            const int global_idx = chunk_base + local_idx;
            float v = 0.0f;
            if (local_idx < CHUNK && global_idx < N) {
                v = x[global_idx];
            }
            const std::uint32_t f = encode_emb(v * inv_s, e, m, is_signed);
            packed |= (f << (k * w));
        }
        o_chunk[word] = packed;
    }
}

static __global__ void decode_chunk_kernel(
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scales,
    float*               __restrict__ out,
    int N,
    int e, int m, int is_signed,
    int npw, int wpc, int w)
{
    const int chunk = blockIdx.x;
    const int lane  = threadIdx.x;
    const int chunk_base = chunk * CHUNK;

    const float s = scales[chunk];
    const std::uint32_t mask = (w == 32) ? 0xFFFFFFFFu : ((1u << w) - 1u);

    const std::uint32_t* i_chunk = in + (size_t)chunk * (size_t)wpc;

    for (int word = lane; word < wpc; word += BLK) {
        const std::uint32_t pck = i_chunk[word];
        const int elem_base = word * npw;
        for (int k = 0; k < npw; ++k) {
            const int local_idx  = elem_base + k;
            const int global_idx = chunk_base + local_idx;
            if (local_idx >= CHUNK || global_idx >= N) break;
            const std::uint32_t field = (pck >> (k * w)) & mask;
            out[global_idx] = decode_emb(field, e, m, is_signed) * s;
        }
    }
}


void launch_encode_chunk(
    const float*    x,
    std::uint32_t*  out,
    float*          scales,
    int             N,
    int             e, int m, int is_signed,
    void*           stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    const int w   = element_width(e, m, is_signed);
    const int npw = n_per_word(w);
    const int wpc = (CHUNK + npw - 1) / npw;
    encode_chunk_kernel<<<n_chunks, BLK, 0, cs>>>(
        x, out, scales, N, e, m, is_signed, npw, wpc, w);
}

void launch_decode_chunk(
    const std::uint32_t* in,
    const float*         scales,
    float*               out,
    int                  N,
    int                  e, int m, int is_signed,
    void*                stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    const int w   = element_width(e, m, is_signed);
    const int npw = n_per_word(w);
    const int wpc = (CHUNK + npw - 1) / npw;
    decode_chunk_kernel<<<n_chunks, BLK, 0, cs>>>(
        in, scales, out, N, e, m, is_signed, npw, wpc, w);
}


}  // namespace qbench_lp
