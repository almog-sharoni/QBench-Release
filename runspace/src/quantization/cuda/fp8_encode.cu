// runspace/src/quantization/cuda/fp8_encode.cu
//
// Phase 0 encode kernels:
//   encode_fp8_emb_chunk  : per-chunk power-of-two scale, chunk_size = 128.
//   encode_fp8_tensor     : single scalar scale (host-computed). Accepts any
//                           N >= 1 via vectorized bulk + scalar tail handler.
//   encode_fp8_channel    : per-channel scales (host-computed). K must be a
//                           multiple of 4.
//
// All three share the encode_fp8_emb device function from fp8_codec.cuh.

#include "fp8_codec.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>


// Forward declarations: defined in fp8_decode.cu.
void launch_decode_fp8_emb_chunk(torch::Tensor, torch::Tensor, torch::Tensor,
                                 int, int, int, int);
void launch_decode_fp8_tensor   (torch::Tensor, torch::Tensor, double,
                                 int, int, int);
void launch_decode_fp8_channel  (torch::Tensor, torch::Tensor, torch::Tensor,
                                 int, int, int);


// ============================================================================
// 1. Per-chunk encode.
// ============================================================================

extern "C" __global__
void encode_fp8_emb_chunk(
    const float* __restrict__ x,
    uint8_t*     __restrict__ out,
    float*       __restrict__ scales,
    int N,
    int chunk_size,
    int e, int m, int b)
{
    const int chunk_id = blockIdx.x;
    const int lane     = threadIdx.x;
    const int base     = chunk_id * chunk_size;

    float4 v = reinterpret_cast<const float4*>(x + base)[lane];
    const float local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                                  fmaxf(fabsf(v.z), fabsf(v.w)));
    const float amax  = warp_amax(local_max);
    const float s     = pow2_floor_nonneg(amax);
    const float inv_s = 1.0f / s;

    if (lane == 0) scales[chunk_id] = s;

    const uint32_t packed =
        uint32_t(encode_fp8_emb(v.x * inv_s, e, m, b))         |
        (uint32_t(encode_fp8_emb(v.y * inv_s, e, m, b)) <<  8) |
        (uint32_t(encode_fp8_emb(v.z * inv_s, e, m, b)) << 16) |
        (uint32_t(encode_fp8_emb(v.w * inv_s, e, m, b)) << 24);
    reinterpret_cast<uint32_t*>(out + base)[lane] = packed;
}

void launch_encode_fp8_emb_chunk(
    torch::Tensor x, torch::Tensor out, torch::Tensor scales,
    int e, int m, int b, int chunk_size)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && scales.is_contiguous());

    const int N = x.numel();
    TORCH_CHECK(chunk_size == 128, "R1: chunk_size must be 128");
    TORCH_CHECK(N % chunk_size == 0);
    TORCH_CHECK(out.numel() == N);
    TORCH_CHECK(scales.numel() == N / chunk_size);

    const dim3 grid(N / chunk_size);
    const dim3 block(32);
    encode_fp8_emb_chunk<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<uint8_t>(), scales.data_ptr<float>(),
        N, chunk_size, e, m, b);
    AT_CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// 2. Per-tensor encode. Scalar scale, vectorized bulk + scalar tail.
//
//    Layout:
//      gid in [0, n4)     handles a full quartet via float4 / uint32.
//      gid == n4 (one thread, only if tail > 0) handles the 1, 2, or 3
//                         leftover scalar elements.
// ============================================================================

extern "C" __global__
void encode_fp8_tensor(
    const float* __restrict__ x,
    uint8_t*     __restrict__ out,
    float        inv_s,
    int          N,
    int e, int m, int b)
{
    const int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4   = N >> 2;                  // N / 4
    const int tail = N & 3;                   // N % 4

    if (gid < n4) {
        const float4 v = reinterpret_cast<const float4*>(x)[gid];
        const uint32_t packed =
            uint32_t(encode_fp8_emb(v.x * inv_s, e, m, b))         |
            (uint32_t(encode_fp8_emb(v.y * inv_s, e, m, b)) <<  8) |
            (uint32_t(encode_fp8_emb(v.z * inv_s, e, m, b)) << 16) |
            (uint32_t(encode_fp8_emb(v.w * inv_s, e, m, b)) << 24);
        reinterpret_cast<uint32_t*>(out)[gid] = packed;
        return;
    }

    if (gid == n4 && tail > 0) {
        const int base = n4 << 2;             // n4 * 4
        if (tail >= 1) out[base + 0] = encode_fp8_emb(x[base + 0] * inv_s, e, m, b);
        if (tail >= 2) out[base + 1] = encode_fp8_emb(x[base + 1] * inv_s, e, m, b);
        if (tail >= 3) out[base + 2] = encode_fp8_emb(x[base + 2] * inv_s, e, m, b);
    }
}

void launch_encode_fp8_tensor(
    torch::Tensor x, torch::Tensor out, double scale,
    int e, int m, int b)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kUInt8);
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous());

    const int N = x.numel();
    TORCH_CHECK(out.numel() == N);
    TORCH_CHECK(scale > 0.0, "encode_fp8_tensor: scale must be positive");
    if (N == 0) return;                       // nothing to do

    const int n4         = N / 4;
    const int tail       = N - n4 * 4;
    const int total_work = n4 + (tail > 0 ? 1 : 0);
    const int block      = 256;
    const int grid       = (total_work + block - 1) / block;
    const float inv_s    = 1.0f / static_cast<float>(scale);

    encode_fp8_tensor<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<uint8_t>(),
        inv_s, N, e, m, b);
    AT_CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// 3. Per-channel encode. Input layout [C, K], K multiple of 4.
// ============================================================================

extern "C" __global__
void encode_fp8_channel(
    const float* __restrict__ x,
    const float* __restrict__ scales,
    uint8_t*     __restrict__ out,
    int C, int K, int k4,
    int e, int m, int b)
{
    const int channel_id     = blockIdx.y;
    const int element_in_ch  = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_in_ch >= k4 || channel_id >= C) return;

    const float inv_s = 1.0f / scales[channel_id];
    const int   base  = channel_id * K;

    const float4 v = reinterpret_cast<const float4*>(x + base)[element_in_ch];
    const uint32_t packed =
        uint32_t(encode_fp8_emb(v.x * inv_s, e, m, b))         |
        (uint32_t(encode_fp8_emb(v.y * inv_s, e, m, b)) <<  8) |
        (uint32_t(encode_fp8_emb(v.z * inv_s, e, m, b)) << 16) |
        (uint32_t(encode_fp8_emb(v.w * inv_s, e, m, b)) << 24);
    reinterpret_cast<uint32_t*>(out + base)[element_in_ch] = packed;
}

void launch_encode_fp8_channel(
    torch::Tensor x, torch::Tensor scales, torch::Tensor out,
    int e, int m, int b)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kUInt8);
    TORCH_CHECK(x.is_contiguous() && scales.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "encode_fp8_channel: x must be 2D (C, K)");

    const int C = x.size(0);
    const int K = x.size(1);
    TORCH_CHECK(K % 4 == 0, "encode_fp8_channel: K must be multiple of 4");
    TORCH_CHECK(scales.numel() == C);
    TORCH_CHECK(out.numel() == C * K);

    const int k4    = K / 4;
    const int block = 256;
    const dim3 grid((k4 + block - 1) / block, C);

    encode_fp8_channel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), scales.data_ptr<float>(), out.data_ptr<uint8_t>(),
        C, K, k4, e, m, b);
    AT_CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// PYBIND
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("encode_fp8_emb_chunk", &launch_encode_fp8_emb_chunk);
    mod.def("decode_fp8_emb_chunk", &launch_decode_fp8_emb_chunk);
    mod.def("encode_fp8_tensor",    &launch_encode_fp8_tensor);
    mod.def("decode_fp8_tensor",    &launch_decode_fp8_tensor);
    mod.def("encode_fp8_channel",   &launch_encode_fp8_channel);
    mod.def("decode_fp8_channel",   &launch_decode_fp8_channel);
}
