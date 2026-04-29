// runspace/src/quantization/cuda/ops_chunk.cu
//
// Chunk-mode codec kernels (chunk_size = 128).  Per-format template
// specialization: the dispatcher at the bottom routes (e, m, is_signed) to
// the matching encode/decode template instantiation.

#include "codec.cuh"
#include "codec_launch.h"
#include "formats.def"

#include <cuda_runtime.h>

namespace qbench_lp {

static constexpr int CHUNK            = 128;
static constexpr int WARP             = 32;
static constexpr int CHUNKS_PER_BLOCK = 4;
static constexpr int BLK              = WARP * CHUNKS_PER_BLOCK;  // 128


template <int E, int M, int IS_SIGNED>
static __global__ void encode_chunk_kernel_t(
    const float*    __restrict__ x,
    std::uint32_t*  __restrict__ out,
    float*          __restrict__ scales,
    int N, int n_chunks)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    constexpr int WPC = (CHUNK + NPW - 1) / NPW;

    const int chunk_in_block = threadIdx.x >> 5;
    const int lane           = threadIdx.x & (WARP - 1);
    const int chunk          = blockIdx.x * CHUNKS_PER_BLOCK + chunk_in_block;
    if (chunk >= n_chunks) return;
    const int chunk_base = chunk * CHUNK;

    float local_max = 0.0f;
    if (chunk_base + CHUNK <= N) {
        const float4 v = reinterpret_cast<const float4*>(x)
                             [(chunk_base >> 2) + lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                          fmaxf(fabsf(v.z), fabsf(v.w)));
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = chunk_base + lane + j * WARP;
            const float v = (idx < N) ? x[idx] : 0.0f;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }
    const float amax  = warp_amax(local_max);
    const float s     = pow2_floor_nonneg(amax);
    const float inv_s = 1.0f / s;

    if (lane == 0) scales[chunk] = s;

    std::uint32_t* o_chunk = out + (size_t)chunk * (size_t)WPC;

    #pragma unroll 1
    for (int word = lane; word < WPC; word += WARP) {
        const int elem_base = word * NPW;
        std::uint32_t packed = 0u;
        #pragma unroll
        for (int k = 0; k < NPW; ++k) {
            const int local_idx  = elem_base + k;
            const int global_idx = chunk_base + local_idx;
            float v = 0.0f;
            if (local_idx < CHUNK && global_idx < N) {
                v = x[global_idx];
            }
            const std::uint32_t f = encode_emb(v * inv_s, E, M, IS_SIGNED);
            packed |= (f << (k * W));
        }
        o_chunk[word] = packed;
    }
}

template <int E, int M, int IS_SIGNED>
static __global__ void decode_chunk_kernel_t(
    const std::uint32_t* __restrict__ in,
    const float*         __restrict__ scales,
    float*               __restrict__ out,
    int N, int n_chunks)
{
    constexpr int W   = (IS_SIGNED ? 1 : 0) + E + M;
    constexpr int NPW = 32 / W;
    constexpr int WPC = (CHUNK + NPW - 1) / NPW;
    constexpr std::uint32_t MASK = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);

    const int chunk_in_block = threadIdx.x >> 5;
    const int lane           = threadIdx.x & (WARP - 1);
    const int chunk          = blockIdx.x * CHUNKS_PER_BLOCK + chunk_in_block;
    if (chunk >= n_chunks) return;
    const int chunk_base = chunk * CHUNK;
    const float s = scales[chunk];

    const std::uint32_t* i_chunk = in + (size_t)chunk * (size_t)WPC;

    #pragma unroll 1
    for (int word = lane; word < WPC; word += WARP) {
        const std::uint32_t pck = i_chunk[word];
        const int elem_base = word * NPW;
        #pragma unroll
        for (int k = 0; k < NPW; ++k) {
            const int local_idx  = elem_base + k;
            const int global_idx = chunk_base + local_idx;
            if (local_idx >= CHUNK || global_idx >= N) break;
            const std::uint32_t field = (pck >> (k * W)) & MASK;
            out[global_idx] = decode_emb(field, E, M, IS_SIGNED) * s;
        }
    }
}


template <int E, int M, int IS_SIGNED>
static void launch_encode_chunk_t(
    const float* x, std::uint32_t* out, float* scales,
    int N, void* stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    const int grid     = (n_chunks + CHUNKS_PER_BLOCK - 1) / CHUNKS_PER_BLOCK;
    encode_chunk_kernel_t<E, M, IS_SIGNED><<<grid, BLK, 0, cs>>>(
        x, out, scales, N, n_chunks);
}

template <int E, int M, int IS_SIGNED>
static void launch_decode_chunk_t(
    const std::uint32_t* in, const float* scales, float* out,
    int N, void* stream)
{
    if (N == 0) return;
    auto cs = static_cast<cudaStream_t>(stream);
    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    const int grid     = (n_chunks + CHUNKS_PER_BLOCK - 1) / CHUNKS_PER_BLOCK;
    decode_chunk_kernel_t<E, M, IS_SIGNED><<<grid, BLK, 0, cs>>>(
        in, scales, out, N, n_chunks);
}


// ----------------------------------------------------------------------------
// Public dispatchers
// ----------------------------------------------------------------------------

#define LP_KEY(E, M, S) (((E) << 6) | ((M) << 1) | ((S) & 1))

void launch_encode_chunk(
    const float*    x,
    std::uint32_t*  out,
    float*          scales,
    int             N,
    int             e, int m, int is_signed,
    void*           stream)
{
    const int key = LP_KEY(e, m, is_signed);
    switch (key) {
    #define CASE(E, M, S) case LP_KEY(E, M, S): \
        launch_encode_chunk_t<E, M, S>(x, out, scales, N, stream); return;
    LP_FORMAT_LIST(CASE)
    #undef CASE
    }
}

void launch_decode_chunk(
    const std::uint32_t* in,
    const float*         scales,
    float*               out,
    int                  N,
    int                  e, int m, int is_signed,
    void*                stream)
{
    const int key = LP_KEY(e, m, is_signed);
    switch (key) {
    #define CASE(E, M, S) case LP_KEY(E, M, S): \
        launch_decode_chunk_t<E, M, S>(in, scales, out, N, stream); return;
    LP_FORMAT_LIST(CASE)
    #undef CASE
    }
}


}  // namespace qbench_lp
