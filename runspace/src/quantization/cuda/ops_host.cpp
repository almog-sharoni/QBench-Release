// runspace/src/quantization/cuda/ops_host.cpp
//
// Plain C++ host side: tensor allocation, shape and stride computation,
// and the pybind module.  Built with g++.

#include "codec_launch.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <tuple>

namespace py = pybind11;

using qbench_lp::ChannelDecodeMeta;
using qbench_lp::CHANNEL_MAX_RANK;
using qbench_lp::n_per_word_em;
using qbench_lp::packed_words_em;
using qbench_lp::element_width;


static inline void check_em(int e, int m, int is_signed, const char* who) {
    TORCH_CHECK(is_signed == 0 || is_signed == 1, who,
                ": is_signed must be 0 or 1");
    const int e_max = is_signed ? 15 : 16;
    const int m_max = is_signed ? 15 : 16;
    TORCH_CHECK(e >= 1 && e <= e_max, who,
                ": e must be in [1, ", e_max, "] for is_signed=", is_signed);
    TORCH_CHECK(m >= 0 && m <= m_max, who,
                ": m must be in [0, ", m_max, "] for is_signed=", is_signed);
    const int w = element_width(e, m, is_signed);
    TORCH_CHECK(w >= 2 && w <= 16, who,
                ": element width signed_bit + e + m must be in [2, 16], got ", w);
}

static inline void* current_stream_ptr() {
    return static_cast<void*>(at::cuda::getCurrentCUDAStream().stream());
}


// ============================================================================
// Tensor mode
// ============================================================================

static std::tuple<torch::Tensor, torch::Tensor>
encode_tensor(torch::Tensor x, int e, int m, bool is_signed)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(),
                "encode_tensor: x must be contiguous CUDA float32");
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "encode_tensor");

    const int     N      = (int)x.numel();
    const int64_t nwords = packed_words_em((int64_t)N, e, m, sgn);
    auto          opts   = x.options();

    auto data  = torch::empty({nwords}, opts.dtype(torch::kInt32));
    auto scale = torch::empty({1},      opts.dtype(torch::kFloat32));

    if (N == 0) {
        scale.fill_(1.0f);
        return {data, scale};
    }

    void* stream = current_stream_ptr();

    // pass-1 emits one float per block; sized to match AMAX_BLK*AMAX_VPT
    // (1024 elements/block) in ops_tensor.cu.
    constexpr int PASS1_PER_BLOCK = 256 * 4;
    const int grid1 = (N + PASS1_PER_BLOCK - 1) / PASS1_PER_BLOCK;
    auto partial = torch::empty({grid1}, opts.dtype(torch::kFloat32));

    qbench_lp::launch_compute_scale_tensor(
        x.data_ptr<float>(),
        partial.data_ptr<float>(),
        scale.data_ptr<float>(),
        N, stream);

    qbench_lp::launch_encode_tensor(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        reinterpret_cast<std::uint32_t*>(data.data_ptr<int32_t>()),
        N, e, m, sgn, stream);

    return {data, scale};
}

static torch::Tensor
decode_tensor(torch::Tensor data, torch::Tensor scale,
              int N, int e, int m, bool is_signed)
{
    TORCH_CHECK(data.is_cuda()  && data.scalar_type()  == torch::kInt32   && data.is_contiguous());
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32 && scale.numel() == 1);
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "decode_tensor");

    const int64_t expected_words = packed_words_em((int64_t)N, e, m, sgn);
    TORCH_CHECK(data.numel() == expected_words,
                "decode_tensor: data.numel() must equal packed_words(N, w)");

    auto out = torch::empty({N}, data.options().dtype(torch::kFloat32));
    if (N == 0) return out;

    qbench_lp::launch_decode_tensor(
        reinterpret_cast<const std::uint32_t*>(data.data_ptr<int32_t>()),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        N, e, m, sgn, current_stream_ptr());
    return out;
}


// ============================================================================
// Chunk mode
// ============================================================================

static std::tuple<torch::Tensor, torch::Tensor>
encode_chunk(torch::Tensor x, int e, int m, bool is_signed)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(),
                "encode_chunk: x must be contiguous CUDA float32");
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "encode_chunk");

    constexpr int CHUNK = 128;
    const int64_t B     = (x.dim() > 1) ? x.size(0) : 1;
    const int64_t M     = (B > 0) ? (x.numel() / B) : 0;
    const int64_t pad   = (CHUNK - (M % CHUNK)) % CHUNK;
    const int64_t M_pad = M + pad;
    const int64_t N_pad = B * M_pad;
    const int n_chunks  = (int)(N_pad / CHUNK);
    const int npw       = n_per_word_em(e, m, sgn);
    const int wpc       = (CHUNK + npw - 1) / npw;
    auto      opts      = x.options();

    auto data   = torch::empty({(int64_t)n_chunks * wpc}, opts.dtype(torch::kInt32));
    auto scales = torch::empty({n_chunks},                opts.dtype(torch::kFloat32));
    if (N_pad == 0) return {data, scales};

    torch::Tensor x_padded;
    if (pad == 0) {
        x_padded = x;
    } else {
        auto x_2d = x.reshape({B, M});
        x_padded  = torch::constant_pad_nd(x_2d, {0, pad}, /*value=*/0.0).contiguous();
    }

    qbench_lp::launch_encode_chunk(
        x_padded.data_ptr<float>(),
        reinterpret_cast<std::uint32_t*>(data.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        (int)N_pad, e, m, sgn, current_stream_ptr());

    return {data, scales};
}

static torch::Tensor
decode_chunk(torch::Tensor data, torch::Tensor scales,
             std::vector<int64_t> original_shape,
             int e, int m, bool is_signed)
{
    TORCH_CHECK(data.is_cuda()   && data.scalar_type()   == torch::kInt32   && data.is_contiguous());
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32 && scales.is_contiguous());
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "decode_chunk");

    constexpr int CHUNK = 128;
    const int nd = (int)original_shape.size();
    TORCH_CHECK(nd >= 1, "decode_chunk: original_shape must have at least one dim");
    int64_t numel = 1;
    for (auto d : original_shape) numel *= d;
    const int64_t B     = (nd > 1) ? original_shape[0] : 1;
    const int64_t M     = (B > 0) ? (numel / B) : 0;
    const int64_t pad   = (CHUNK - (M % CHUNK)) % CHUNK;
    const int64_t M_pad = M + pad;
    const int64_t N_pad = B * M_pad;
    const int n_chunks  = (int)(N_pad / CHUNK);
    const int npw       = n_per_word_em(e, m, sgn);
    const int wpc       = (CHUNK + npw - 1) / npw;
    TORCH_CHECK(data.numel()   == (int64_t)n_chunks * wpc,
                "decode_chunk: data.numel() must equal n_chunks * wpc");
    TORCH_CHECK(scales.numel() == n_chunks,
                "decode_chunk: scales.numel() must equal n_chunks");
    TORCH_CHECK(N_pad < ((int64_t)1 << 31),
                "decode_chunk: padded element count exceeds 2^31");

    if (numel == 0) {
        return torch::empty(original_shape, data.options().dtype(torch::kFloat32));
    }

    if (pad == 0) {
        auto out = torch::empty(original_shape, data.options().dtype(torch::kFloat32));
        qbench_lp::launch_decode_chunk(
            reinterpret_cast<const std::uint32_t*>(data.data_ptr<int32_t>()),
            scales.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)N_pad, e, m, sgn, current_stream_ptr());
        return out;
    }

    auto padded = torch::empty({N_pad}, data.options().dtype(torch::kFloat32));
    qbench_lp::launch_decode_chunk(
        reinterpret_cast<const std::uint32_t*>(data.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        padded.data_ptr<float>(),
        (int)N_pad, e, m, sgn, current_stream_ptr());
    return padded.reshape({B, M_pad}).slice(1, 0, M).contiguous().reshape(original_shape);
}


// ============================================================================
// Channel mode
// ============================================================================

static std::tuple<torch::Tensor, torch::Tensor>
encode_channel(torch::Tensor x, int channel_dim, int e, int m, bool is_signed)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(),
                "encode_channel: x must be contiguous CUDA float32");
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "encode_channel");
    const int nd = (int)x.dim();
    TORCH_CHECK(nd >= 1, "encode_channel: x must have at least one dim");
    TORCH_CHECK(channel_dim >= 0 && channel_dim < nd,
                "encode_channel: channel_dim out of range");

    std::vector<int64_t> perm;
    perm.reserve(nd);
    perm.push_back((int64_t)channel_dim);
    for (int64_t d = 0; d < nd; ++d)
        if ((int)d != channel_dim) perm.push_back(d);

    auto x_perm = (nd > 1) ? x.permute(perm).contiguous() : x.contiguous();
    const int C = (int)x_perm.size(0);
    const int K = (int)(x_perm.numel() / std::max(C, 1));

    const int npw   = n_per_word_em(e, m, sgn);
    const int K_pad = ((K + npw - 1) / npw) * npw;
    const int wpr   = K_pad / npw;
    auto      opts  = x.options();

    auto data   = torch::empty({C, wpr}, opts.dtype(torch::kInt32));
    auto scales = torch::empty({C},      opts.dtype(torch::kFloat32));
    if (C == 0 || K == 0) return {data, scales};

    auto x_2d = x_perm.reshape({C, K}).contiguous();
    torch::Tensor xc;
    if (K_pad == K) {
        xc = x_2d;
    } else {
        xc = torch::zeros({C, K_pad}, opts);
        xc.slice(1, 0, K).copy_(x_2d);
        xc = xc.contiguous();
    }

    qbench_lp::launch_encode_channel(
        xc.data_ptr<float>(),
        reinterpret_cast<std::uint32_t*>(data.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        C, K_pad, e, m, sgn, current_stream_ptr());

    return {data, scales};
}

static torch::Tensor
decode_channel(torch::Tensor data, torch::Tensor scales,
               std::vector<int64_t> original_shape, int channel_dim,
               int e, int m, bool is_signed)
{
    TORCH_CHECK(data.is_cuda() && data.dim() == 2
                && data.scalar_type() == torch::kInt32 && data.is_contiguous(),
                "decode_channel: data must be contiguous CUDA int32 of shape [C, words_per_row]");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32,
                "decode_channel: scales must be CUDA float32");
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "decode_channel");

    const int nd = (int)original_shape.size();
    TORCH_CHECK(nd >= 1 && nd <= CHANNEL_MAX_RANK,
                "decode_channel: original_shape rank must be in [1, CHANNEL_MAX_RANK]");
    TORCH_CHECK(channel_dim >= 0 && channel_dim < nd,
                "decode_channel: channel_dim out of range");

    const int C = (int)original_shape[channel_dim];
    int64_t numel = 1;
    for (auto d : original_shape) numel *= d;
    const int K     = (C > 0) ? (int)(numel / C) : 0;
    const int npw   = n_per_word_em(e, m, sgn);
    const int K_pad = ((K + npw - 1) / npw) * npw;
    const int wpr   = K_pad / npw;

    TORCH_CHECK((int)data.size(0) == C, "decode_channel: data.size(0) != C");
    TORCH_CHECK(scales.numel() == C,    "decode_channel: scales.numel() != C");
    TORCH_CHECK((int)data.size(1) == wpr,
                "decode_channel: data.size(1) inconsistent with original_shape and (e, m, is_signed)");
    TORCH_CHECK(numel < ((int64_t)1 << 31),
                "decode_channel: numel exceeds 2^31; int32 indexing insufficient");

    auto out = torch::empty(original_shape, data.options().dtype(torch::kFloat32));
    if (numel == 0 || C == 0 || K == 0) return out;

    ChannelDecodeMeta meta{};
    meta.nd          = nd;
    meta.channel_dim = channel_dim;

    int64_t s = 1;
    for (int d = nd - 1; d >= 0; --d) {
        meta.stride_orig[d] = (int)s;
        s *= original_shape[d];
    }

    int n_other = 0;
    for (int d = 0; d < nd; ++d) {
        if (d == channel_dim) continue;
        meta.other_axes [n_other] = d;
        meta.other_sizes[n_other] = (int)original_shape[d];
        ++n_other;
    }
    meta.n_other = n_other;

    int64_t acc = 1;
    for (int j = n_other - 1; j >= 0; --j) {
        meta.inner_stride[j] = (int)acc;
        acc *= meta.other_sizes[j];
    }

    qbench_lp::launch_decode_channel(
        reinterpret_cast<const std::uint32_t*>(data.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        out.data_ptr<float>(),
        C, K, K_pad, e, m, sgn, meta, current_stream_ptr());

    return out;
}


// ============================================================================
// Round-trip helpers: encode + decode + scale_b/scale_p/max_val construction
// ============================================================================
//
// These collapse the per-mode Python wrapper in `_quantize_tensor_cuda`
// (runspace/src/ops/quant_base.py) into C++.  Each returns the four tensors
// the Python tail consumes: input_fp8 (in original shape), scale_b
// (broadcast scale, shape == input.shape), scale_p (packed scale), max_val.

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
roundtrip_tensor(torch::Tensor x, int e, int m, bool is_signed)
{
    auto enc       = encode_tensor(x, e, m, is_signed);
    auto& data     = std::get<0>(enc);
    auto& scale    = std::get<1>(enc);
    auto input_fp8 = decode_tensor(data, scale, (int)x.numel(),
                                   e, m, is_signed).reshape_as(x);
    torch::Tensor scale_b;
    if (x.dim() > 0) {
        std::vector<int64_t> view_shape((size_t)x.dim(), (int64_t)1);
        scale_b = scale.view(view_shape);
    } else {
        scale_b = scale;
    }
    return {input_fp8, scale_b, scale_b, scale_b};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
roundtrip_chunk(torch::Tensor x, int e, int m, bool is_signed,
                bool needs_scale_b)
{
    constexpr int CHUNK = 128;
    const int64_t B = (x.dim() > 1) ? x.size(0) : 1;
    const int64_t M = (B > 0) ? (x.numel() / B) : 0;
    const int64_t n_chunks_per_batch = (M + CHUNK - 1) / CHUNK;

    auto enc       = encode_chunk(x, e, m, is_signed);
    auto& data     = std::get<0>(enc);
    auto& scales   = std::get<1>(enc);
    auto input_fp8 = decode_chunk(data, scales, x.sizes().vec(),
                                  e, m, is_signed);
    auto scale_p   = scales.view({B, n_chunks_per_batch, (int64_t)1});

    torch::Tensor scale_b;
    if (needs_scale_b) {
        auto expanded = scale_p.expand({B, n_chunks_per_batch, (int64_t)CHUNK})
                               .contiguous()
                               .reshape({B, n_chunks_per_batch * CHUNK});
        if (expanded.size(-1) != M) {
            expanded = expanded.slice(1, 0, M).contiguous();
        }
        scale_b = expanded.reshape(x.sizes());
    } else {
        scale_b = scale_p;
    }

    auto max_val = scale_p.max();
    return {input_fp8, scale_b, scale_p, max_val};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
roundtrip_channel(torch::Tensor x, int channel_dim,
                  int e, int m, bool is_signed)
{
    auto enc       = encode_channel(x, channel_dim, e, m, is_signed);
    auto& data     = std::get<0>(enc);
    auto& scales   = std::get<1>(enc);
    auto input_fp8 = decode_channel(data, scales, x.sizes().vec(),
                                    channel_dim, e, m, is_signed);
    std::vector<int64_t> view_shape((size_t)x.dim(), (int64_t)1);
    view_shape[channel_dim] = x.size(channel_dim);
    auto scale_b = scales.view(view_shape);
    return {input_fp8, scale_b, scale_b, scale_b};
}


// ============================================================================
// Helpers exposed to Python
// ============================================================================

static int n_per_word_py(int e, int m, bool is_signed) {
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "n_per_word");
    return n_per_word_em(e, m, sgn);
}

static int element_width_py(int e, int m, bool is_signed) {
    const int sgn = is_signed ? 1 : 0;
    check_em(e, m, sgn, "element_width");
    return element_width(e, m, sgn);
}


// ============================================================================
// Pybind module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("encode_tensor",  &encode_tensor,
            py::arg("x"), py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Tensor-mode encode. Returns (data int32[packed_words], scale float32[1]).");
    mod.def("decode_tensor",  &decode_tensor,
            py::arg("data"), py::arg("scale"),
            py::arg("N"), py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Tensor-mode decode. Returns float32[N].");

    mod.def("encode_chunk",   &encode_chunk,
            py::arg("x"), py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Chunk-mode encode (chunk_size = 128).");
    mod.def("decode_chunk",   &decode_chunk,
            py::arg("data"), py::arg("scales"),
            py::arg("original_shape"),
            py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Chunk-mode decode. Returns float32 tensor in `original_shape` "
            "(per-batch padding is truncated internally).");

    mod.def("encode_channel", &encode_channel,
            py::arg("x"), py::arg("channel_dim"),
            py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Channel-mode encode.");
    mod.def("decode_channel", &decode_channel,
            py::arg("data"), py::arg("scales"),
            py::arg("original_shape"), py::arg("channel_dim"),
            py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Channel-mode decode. Returns float32 tensor in `original_shape`.");

    mod.def("roundtrip_tensor",  &roundtrip_tensor,
            py::arg("x"), py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Tensor-mode round trip. Returns (input_fp8, scale_b, scale_p, max_val).");
    mod.def("roundtrip_chunk",   &roundtrip_chunk,
            py::arg("x"), py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            py::arg("needs_scale_b") = true,
            "Chunk-mode round trip. Returns (input_fp8, scale_b, scale_p, max_val). "
            "When needs_scale_b is false, scale_b aliases scale_p (no input-shaped allocation).");
    mod.def("roundtrip_channel", &roundtrip_channel,
            py::arg("x"), py::arg("channel_dim"),
            py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Channel-mode round trip. Returns (input_fp8, scale_b, scale_p, max_val).");

    mod.def("n_per_word",     &n_per_word_py,
            py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Number of (e, m) elements packed into one uint32 storage word.");
    mod.def("element_width",  &element_width_py,
            py::arg("e"), py::arg("m"),
            py::arg("is_signed") = true,
            "Element width in bits, signed_bit + e + m.");
}
