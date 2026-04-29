// runspace/src/quantization/cuda/codec.cuh
//
// Generic low-precision floating-point codec primitives.  Element width
//
//     w = signed_bit + e + m,           2 <= w <= 16
//
// where signed_bit is 1 for signed formats and 0 for unsigned formats.
//
// Signed layout (signed_bit = 1):
//
//      bit  w-1     w-2 ... m       m-1 ... 0
//          +---+---------------+---------------+
//          | S |   exponent    |    mantissa   |
//          +---+---------------+---------------+
//
// Unsigned layout (signed_bit = 0):
//
//      bit  w-1 ... m       m-1 ... 0
//          +---------------+---------------+
//          |   exponent    |    mantissa   |
//          +---------------+---------------+
//
// In unsigned mode every encoded value is nonnegative.  Negative inputs
// flush to +0 in encode; this matches the "treat as zero" convention of
// unsigned integer types and keeps the codec's existing zero-handling
// path intact.  All other algorithmic steps are unchanged.
//
// Bias convention:
//   The codec is invoked on y = x / s, with s = 2^floor(log2(amax(x))).
//   The implicit byte-format bias is b = 2^e - 1, fixed by e.  For
//   y in [-2, 2) the unbiased FP32 exponent satisfies exp_f <= 0, and
//   the encoded exponent field is exp_t = exp_f + b in [0, 2^e - 1].
//
// Numerical algorithm:
//   This is a bit-exact translation of `quantize_fp_generic` from
//   runspace/src/quantization/quantizer.py for the signed case; the
//   unsigned case is the same algorithm with the sign bit removed and
//   negative inputs flushed.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>


// ----------------------------------------------------------------------------
// pow2_floor_nonneg(a) -- s = 2^floor(log2(a)) by clearing the FP32 mantissa.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float pow2_floor_nonneg(float a) {
    if (a == 0.0f) return 1.0f;
    uint32_t bits = __float_as_uint(a);
    bits &= 0xFF800000u;
    return __uint_as_float(bits);
}


// ----------------------------------------------------------------------------
// warp_amax(v) -- butterfly reduction over |v| across 32 warp lanes.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float warp_amax(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    }
    return v;
}


// ----------------------------------------------------------------------------
// encode_emb(y, e, m, is_signed) -- FP32 -> w-bit field in a uint32.
//
// Output occupies the low (is_signed + e + m) bits of the returned
// uint32.  In signed mode the sign bit sits at position (e + m); in
// unsigned mode there is no sign bit and the layout is just
// (exp << m) | mant.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint32_t encode_emb(float y, int e, int m, int is_signed) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    // Unsigned mode: any negative input (including -0) flushes to +0.
    if (!is_signed && sign) {
        return 0u;
    }

    if (mag == 0u) {
        // Signed zero (or +0 in unsigned mode).
        return is_signed ? (sign << (e + m)) : 0u;
    }

    const int      b        = (1 << e) - 1;
    const uint32_t max_exp  = (uint32_t)b;
    const uint32_t max_mant = (m == 0) ? 0u : ((1u << m) - 1u);

    int32_t  exp_f     = (int32_t)((mag >> 23) & 0xFFu) - 127;
    uint32_t mant_full = (mag & 0x7FFFFFu) | (1u << 23);

    int m_mask = (1 - b) - exp_f;
    if (m_mask < 0) m_mask = 0;
    int shift = (23 - m) + m_mask;

    uint32_t round_bit = 0u;
    if (shift >= 1 && shift <= 24) {
        round_bit = (mant_full >> (shift - 1)) & 1u;
    }

    uint32_t mant_trunc;
    if (shift >= 24) {
        mant_trunc = 0u;
    } else {
        mant_trunc = (mant_full >> shift) << shift;
    }
    if (round_bit) {
        mant_trunc += (1u << shift);
    }

    const uint32_t overflow = (mant_trunc >> 24) & 1u;
    if (overflow) exp_f += 1;

    const uint32_t bits_23_24 = (mant_trunc >> 23) & 3u;
    if (bits_23_24 == 0u) {
        return is_signed ? (sign << (e + m)) : 0u;
    }

    const uint32_t sign_field = is_signed ? (sign << (e + m)) : 0u;

    if (exp_f > 0) {
        return sign_field | (max_exp << m) | max_mant;
    }

    uint32_t exp_t;
    if (exp_f >= 1 - b) {
        exp_t = (uint32_t)(exp_f + b);
    } else {
        exp_t = 0u;
    }
    const uint32_t mant_t = (m == 0) ? 0u
                                     : ((mant_trunc >> shift) & max_mant);

    return sign_field | (exp_t << m) | mant_t;
}


// ----------------------------------------------------------------------------
// decode_emb(field, e, m, is_signed) -- w-bit field -> FP32.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float decode_emb(uint32_t field, int e, int m, int is_signed) {
    const uint32_t sign  = is_signed ? ((field >> (e + m)) & 1u) : 0u;
    const uint32_t exp_t = (field >> m) & ((1u << e) - 1u);
    const uint32_t mant  = (m == 0) ? 0u : (field & ((1u << m) - 1u));

    const int b = (1 << e) - 1;

    if (exp_t == 0u) {
        if (mant == 0u) return sign ? -0.0f : 0.0f;
        const float v = ldexpf((float)mant, 1 - b - m);
        return sign ? -v : v;
    }

    const int32_t  exp_f  = (int32_t)exp_t - b + 127;
    const uint32_t mant_f = mant << (23 - m);
    const uint32_t bits   = (sign << 31) | ((uint32_t)exp_f << 23) | mant_f;
    return __uint_as_float(bits);
}
