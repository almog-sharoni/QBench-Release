// runspace/src/quantization/cuda/codec_launch.h
//
// Host-side launch declarations and packing geometry helpers.

#pragma once

#include <cstdint>
#include <cstddef>


namespace qbench_lp {


// ----------------------------------------------------------------------------
// Packing geometry
// ----------------------------------------------------------------------------
//
// Element width is
//
//     w = signed_bit + e + m,           2 <= w <= 16
//
// where signed_bit is 1 for signed types and 0 for unsigned.  Each
// uint32 storage word holds n_per_word(w) = floor(32 / w) elements at
// lane offsets 0, w, 2w, ...; trailing low bits are zero padding.

constexpr int element_width(int e, int m, int is_signed) {
    return (is_signed ? 1 : 0) + e + m;
}
constexpr int n_per_word(int w)               { return 32 / w; }
constexpr int n_per_word_em(int e, int m, int is_signed) {
    return n_per_word(element_width(e, m, is_signed));
}

constexpr std::int64_t packed_words(std::int64_t N, int w) {
    const int npw = n_per_word(w);
    return (N + npw - 1) / npw;
}
constexpr std::int64_t packed_words_em(std::int64_t N, int e, int m, int is_signed) {
    return packed_words(N, element_width(e, m, is_signed));
}


// ----------------------------------------------------------------------------
// Tensor mode
// ----------------------------------------------------------------------------

// partial must hold one float per pass-1 block; see encode_tensor in
// ops_host.cpp for the host-side sizing.
void launch_compute_scale_tensor(
    const float* x,
    float*       partial,
    float*       scale_out,
    int          N,
    void*        stream);

void launch_encode_tensor(
    const float*    x,
    const float*    scale_ptr,
    std::uint32_t*  out,
    int             N,
    int             e, int m, int is_signed,
    void*           stream);

void launch_decode_tensor(
    const std::uint32_t* in,
    const float*         scale_ptr,
    float*               out,
    int                  N,
    int                  e, int m, int is_signed,
    void*                stream);


// ----------------------------------------------------------------------------
// Chunk mode (chunk_size = 128)
// ----------------------------------------------------------------------------

void launch_encode_chunk(
    const float*    x,
    std::uint32_t*  out,
    float*          scales,
    int             N,
    int             e, int m, int is_signed,
    void*           stream);

void launch_decode_chunk(
    const std::uint32_t* in,
    const float*         scales,
    float*               out,
    int                  N,
    int                  e, int m, int is_signed,
    void*                stream);


// ----------------------------------------------------------------------------
// Channel mode
// ----------------------------------------------------------------------------

constexpr int CHANNEL_MAX_RANK = 8;

struct ChannelDecodeMeta {
    int  nd;
    int  channel_dim;
    int  n_other;
    int  other_axes  [CHANNEL_MAX_RANK];
    int  other_sizes [CHANNEL_MAX_RANK];
    int  inner_stride[CHANNEL_MAX_RANK];
    int  stride_orig [CHANNEL_MAX_RANK];
};

void launch_encode_channel(
    const float*    x,
    std::uint32_t*  out,
    float*          scales,
    int             C, int K_pad,
    int             e, int m, int is_signed,
    void*           stream);

void launch_decode_channel(
    const std::uint32_t*       in,
    const float*               scales,
    float*                     out,
    int                        C, int K, int K_pad,
    int                        e, int m, int is_signed,
    const ChannelDecodeMeta&   meta,
    void*                      stream);


}  // namespace qbench_lp
