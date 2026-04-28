// runspace/src/quantization/cuda/fp8_codec.cuh
#pragma once

#include <cuda_runtime.h>
#include <cstdint>


// ----------------------------------------------------------------------------
// Helper A: pow2_floor_nonneg.
// Returns 2^floor(log2(a)) for a > 0 by zeroing the FP32 mantissa.
// For a == 0 returns 1.0f as a safe neutral scale.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float pow2_floor_nonneg(float a) {
    if (a == 0.0f) return 1.0f;
    uint32_t bits = __float_as_uint(a);
    bits &= 0xFF800000u;
    return __uint_as_float(bits);
}


// ----------------------------------------------------------------------------
// Helper B: warp_amax. Butterfly reduction over all 32 lanes; result on every lane.
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
// Helper C: encode_fp8. FP32 -> FP8(e, m, b) with RNTE and permissive saturation.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8(float y, int e, int m, int b) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag == 0u) return 0u;

    const int32_t  exp_f  = int32_t((mag >> 23) & 0xFFu) - 127;
    const uint32_t mant_f = mag & 0x7FFFFFu;
    const int32_t  exp_t  = exp_f + b;

    if (exp_t < 1) return uint8_t(sign << 7);                 // R6 simplified flush

    const int      shift     = 23 - m;
    const uint32_t round_bit = (mant_f >> (shift - 1)) & 1u;
    const uint32_t sticky    = (mant_f & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
    uint32_t       mant_t    = mant_f >> shift;
    const uint32_t round_up  = round_bit & (sticky | (mant_t & 1u));
    mant_t += round_up;

    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) {
        mant_t  = 0u;
        exp_out += 1;
    }

    // Permissive saturation: only clamp on actual overflow. exp_out == max_exp
    // is a legal normal (matches QBench reference and OCP E4M3 permissive).
    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) {                                   // FIX: > not >=
        exp_out = max_exp;
        mant_t  = (1u << m) - 1u;
    }

    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper C2: encode_fp8_ARU. Same as encode_fp8 but Always Rounds Up (no sticky).
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_ARU(float y, int e, int m, int b) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag == 0u) return 0u;

    const int32_t  exp_f  = int32_t((mag >> 23) & 0xFFu) - 127;
    const uint32_t mant_f = mag & 0x7FFFFFu;
    const int32_t  exp_t  = exp_f + b;

    if (exp_t < 1) return uint8_t(sign << 7);

    const int      shift     = 23 - m;
    const uint32_t round_bit = (mant_f >> (shift - 1)) & 1u;
    uint32_t       mant_t    = mant_f >> shift;
    mant_t += round_bit;                                        // ARU: no sticky

    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) {
        mant_t  = 0u;
        exp_out += 1;
    }

    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) {
        exp_out = max_exp;
        mant_t  = (1u << m) - 1u;
    }

    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper C3: encode_fp8_nf. Same as encode_fp8 (RNTE) but No subnormal Flush.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_nf(float y, int e, int m, int b) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag == 0u) return 0u;

    const int32_t  exp_f  = int32_t((mag >> 23) & 0xFFu) - 127;
    const uint32_t mant_f = mag & 0x7FFFFFu;
    const int32_t  exp_t  = exp_f + b;

    // No flush guard: values with exp_t < 1 fall through the rounding path.

    const int      shift     = 23 - m;
    const uint32_t round_bit = (mant_f >> (shift - 1)) & 1u;
    const uint32_t sticky    = (mant_f & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
    uint32_t       mant_t    = mant_f >> shift;
    const uint32_t round_up  = round_bit & (sticky | (mant_t & 1u));
    mant_t += round_up;

    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) {
        mant_t  = 0u;
        exp_out += 1;
    }

    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) {
        exp_out = max_exp;
        mant_t  = (1u << m) - 1u;
    }

    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper C4: encode_fp8_ARU_nf. ARU rounding + No subnormal Flush.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_ARU_nf(float y, int e, int m, int b) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag == 0u) return 0u;

    const int32_t  exp_f  = int32_t((mag >> 23) & 0xFFu) - 127;
    const uint32_t mant_f = mag & 0x7FFFFFu;
    const int32_t  exp_t  = exp_f + b;

    // No flush guard.

    const int      shift     = 23 - m;
    const uint32_t round_bit = (mant_f >> (shift - 1)) & 1u;
    uint32_t       mant_t    = mant_f >> shift;
    mant_t += round_bit;                                        // ARU: no sticky

    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) {
        mant_t  = 0u;
        exp_out += 1;
    }

    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) {
        exp_out = max_exp;
        mant_t  = (1u << m) - 1u;
    }

    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper D: decode_fp8. FP8(e, m, b) -> FP32. Inverse of encode_fp8.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float decode_fp8(uint8_t byte, int e, int m, int b) {
    const uint32_t u     = uint32_t(byte);
    const uint32_t sign  = (u >> 7) & 1u;
    const uint32_t exp_t = (u >> m) & ((1u << e) - 1u);
    const uint32_t mant  = u & ((1u << m) - 1u);

    if (exp_t == 0u) {
        if (mant == 0u) return sign ? -0.0f : 0.0f;
        const float v = ldexpf(float(mant), 1 - b - m);
        return sign ? -v : v;
    }

    const int32_t  exp_f  = int32_t(exp_t) - b + 127;
    const uint32_t mant_f = mant << (23 - m);
    const uint32_t bits   = (sign << 31) | (uint32_t(exp_f) << 23) | mant_f;
    return __uint_as_float(bits);
}
