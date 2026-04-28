// runspace/src/quantization/cuda/fp8_decode.cu
//
// Phase 0 decode kernels:
//   decode_fp8_chunk : per-chunk power-of-two scale.
//   decode_fp8_tensor    : single scalar scale. Accepts any N >= 1 via
//                          vectorized bulk + scalar tail handler.
//   decode_fp8_channel   : per-channel scales, K multiple of 4.

#include "fp8_codec.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>


// ============================================================================
// 1. Per-chunk decode.
// ============================================================================

extern "C" __global__
void decode_fp8_chunk(
    const uint8_t* __restrict__ in,
    const float*   __restrict__ scales,
    float*         __restrict__ out,
    int N, int chunk_size,
    int e, int m, int b)
{
    const int chunk_id = blockIdx.x;
    const int lane     = threadIdx.x;
    const int base     = chunk_id * chunk_size;

    const float    s      = scales[chunk_id];
    const uint32_t packed = reinterpret_cast<const uint32_t*>(in + base)[lane];

    float4 v;
    v.x = decode_fp8(uint8_t( packed        & 0xFFu), e, m, b) * s;
    v.y = decode_fp8(uint8_t((packed >>  8) & 0xFFu), e, m, b) * s;
    v.z = decode_fp8(uint8_t((packed >> 16) & 0xFFu), e, m, b) * s;
    v.w = decode_fp8(uint8_t((packed >> 24) & 0xFFu), e, m, b) * s;

    reinterpret_cast<float4*>(out + base)[lane] = v;
}

void launch_decode_fp8_chunk(
    torch::Tensor in, torch::Tensor scales, torch::Tensor out,
    int e, int m, int b, int chunk_size)
{
    TORCH_CHECK(in.is_cuda() && in.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32);
    TORCH_CHECK(in.is_contiguous() && scales.is_contiguous() && out.is_contiguous());

    const int N = in.numel();
    TORCH_CHECK(chunk_size == 128, "R1: chunk_size must be 128");
    TORCH_CHECK(N % chunk_size == 0);
    TORCH_CHECK(out.numel() == N);
    TORCH_CHECK(scales.numel() == N / chunk_size);

    const dim3 grid(N / chunk_size);
    const dim3 block(32);
    decode_fp8_chunk<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.data_ptr<uint8_t>(), scales.data_ptr<float>(), out.data_ptr<float>(),
        N, chunk_size, e, m, b);
    AT_CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// 2. Per-tensor decode. Scalar scale, vectorized bulk + scalar tail.
// ============================================================================

extern "C" __global__
void decode_fp8_tensor(
    const uint8_t* __restrict__ in,
    float*         __restrict__ out,
    float          s,
    int            N,
    int e, int m, int b)
{
    const int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4   = N >> 2;
    const int tail = N & 3;

    if (gid < n4) {
        const uint32_t packed = reinterpret_cast<const uint32_t*>(in)[gid];
        float4 v;
        v.x = decode_fp8(uint8_t( packed        & 0xFFu), e, m, b) * s;
        v.y = decode_fp8(uint8_t((packed >>  8) & 0xFFu), e, m, b) * s;
        v.z = decode_fp8(uint8_t((packed >> 16) & 0xFFu), e, m, b) * s;
        v.w = decode_fp8(uint8_t((packed >> 24) & 0xFFu), e, m, b) * s;
        reinterpret_cast<float4*>(out)[gid] = v;
        return;
    }

    if (gid == n4 && tail > 0) {
        const int base = n4 << 2;
        if (tail >= 1) out[base + 0] = decode_fp8(in[base + 0], e, m, b) * s;
        if (tail >= 2) out[base + 1] = decode_fp8(in[base + 1], e, m, b) * s;
        if (tail >= 3) out[base + 2] = decode_fp8(in[base + 2], e, m, b) * s;
    }
}

void launch_decode_fp8_tensor(
    torch::Tensor in, torch::Tensor out, double scale,
    int e, int m, int b)
{
    TORCH_CHECK(in.is_cuda() && in.scalar_type() == torch::kUInt8);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32);
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous());

    const int N = in.numel();
    TORCH_CHECK(out.numel() == N);
    if (N == 0) return;

    const int n4         = N / 4;
    const int tail       = N - n4 * 4;
    const int total_work = n4 + (tail > 0 ? 1 : 0);
    const int block      = 256;
    const int grid       = (total_work + block - 1) / block;
    const float s        = static_cast<float>(scale);

    decode_fp8_tensor<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.data_ptr<uint8_t>(), out.data_ptr<float>(),
        s, N, e, m, b);
    AT_CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// 3. Per-channel decode.
// ============================================================================

extern "C" __global__
void decode_fp8_channel(
    const uint8_t* __restrict__ in,
    const float*   __restrict__ scales,
    float*         __restrict__ out,
    int C, int K, int k4,
    int e, int m, int b)
{
    const int channel_id    = blockIdx.y;
    const int element_in_ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_in_ch >= k4 || channel_id >= C) return;

    const float s    = scales[channel_id];
    const int   base = channel_id * K;

    const uint32_t packed = reinterpret_cast<const uint32_t*>(in + base)[element_in_ch];
    float4 v;
    v.x = decode_fp8(uint8_t( packed        & 0xFFu), e, m, b) * s;
    v.y = decode_fp8(uint8_t((packed >>  8) & 0xFFu), e, m, b) * s;
    v.z = decode_fp8(uint8_t((packed >> 16) & 0xFFu), e, m, b) * s;
    v.w = decode_fp8(uint8_t((packed >> 24) & 0xFFu), e, m, b) * s;
    reinterpret_cast<float4*>(out + base)[element_in_ch] = v;
}

void launch_decode_fp8_channel(
    torch::Tensor in, torch::Tensor scales, torch::Tensor out,
    int e, int m, int b)
{
    TORCH_CHECK(in.is_cuda() && in.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32);
    TORCH_CHECK(in.is_contiguous() && scales.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(out.dim() == 2, "decode_fp8_channel: out must be 2D (C, K)");

    const int C = out.size(0);
    const int K = out.size(1);
    TORCH_CHECK(K % 4 == 0);
    TORCH_CHECK(in.numel() == C * K);
    TORCH_CHECK(scales.numel() == C);

    const int k4    = K / 4;
    const int block = 256;
    const dim3 grid((k4 + block - 1) / block, C);

    decode_fp8_channel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.data_ptr<uint8_t>(), scales.data_ptr<float>(), out.data_ptr<float>(),
        C, K, k4, e, m, b);
    AT_CUDA_CHECK(cudaGetLastError());
}
