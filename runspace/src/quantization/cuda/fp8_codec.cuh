// runspace/src/quantization/cuda/fp8_codec.cuh
//
// Shared device helpers for FP8(e, m, b) codecs used by the QBench Phase 0
// encode and decode kernels and by the encode epilogue of every compute
// kernel in the plan §2.2 inventory.
//
// Conventions (see plan §3.3 risk R4 and §6 risk R6):
//   - FP8 layout per element: 1 sign | e exponent | m mantissa, with
//     1 + e + m == 8. Biased exponent uses the runtime argument b.
//   - Rounding: round to nearest, ties to even (IEEE 754-2008 §4.3.1).
//   - Saturation: clamp magnitude only when exp_out > max_exp = (1 << e) - 1.
//     Permissive convention; treats every bit pattern with exp == max_exp as
//     a legal normal. Matches QBench quantize_fp_generic and OCP E4M3
//     permissive. For OCP E5M2 strict (reserve all-ones exp for Inf/NaN),
//     pass an effective `e' = e - 1` cap by the caller, or post-mask.
//   - Subnormals: flushed to zero below 2^(1 - b). Matches QBench BUG-001
//     in the existing Python quantizer per R6.
//
// References
//   [1] IEEE Std 754-2008, §3.4, §4.3.1.
//   [2] NVIDIA, CUDA C++ Programming Guide, §B.20 Warp Shuffle Functions.
//   [3] OCP, 8-bit Floating Point Specification (OFP8) Revision 1.0, §3, §4.

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
// Helper C: encode_fp8_emb. FP32 -> FP8(e, m, b) with RNTE and permissive saturation.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_emb(float y, int e, int m, int b) {
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
// Helper C2: encode_fp8_emb_rhup. FP32 -> FP8(e, m, b) with round-half-up.
// Identical to encode_fp8_emb except ties always round up (no sticky/lsb check).
// Matches the Python quantize_fp_generic fallback rounding convention.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_emb_rhup(float y, int e, int m, int b) {
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
    mant_t += round_bit;                              // round-half-up: always round up on tie

    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) { mant_t = 0u; exp_out += 1; }

    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) { exp_out = max_exp; mant_t = (1u << m) - 1u; }

    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper C3: encode_fp8_emb_noflush. RNTE + proper subnormal encoding.
// Values below 2^(1-b) are encoded as FP8 subnormals instead of flushed to 0.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_emb_noflush(float y, int e, int m, int b) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag == 0u) return 0u;

    const int32_t  exp_f  = int32_t((mag >> 23) & 0xFFu) - 127;
    const uint32_t mant_f = mag & 0x7FFFFFu;
    const int32_t  exp_t  = exp_f + b;

    if (exp_t < 1) {
        // Subnormal: stored_exp=0, mant = round(|y| * 2^(b+m-1))
        // sub_shift = bits to right-shift the FP32 significand to get FP8 mant.
        const int sub_shift = (23 - m) + (1 - exp_t);
        if (sub_shift > 24) return uint8_t(sign << 7);     // below min subnormal
        const uint32_t full_mant = (1u << 23) | mant_f;
        const uint32_t round_bit = (full_mant >> (sub_shift - 1)) & 1u;
        const uint32_t sticky    = (full_mant & ((1u << (sub_shift - 1)) - 1u)) ? 1u : 0u;
        uint32_t mant_t = full_mant >> sub_shift;
        mant_t += round_bit & (sticky | (mant_t & 1u));    // RNTE
        if (mant_t == (1u << m))
            return uint8_t((sign << 7) | (1u << m));       // overflow to smallest normal
        return uint8_t((sign << 7) | mant_t);
    }

    // Normal range (identical to encode_fp8_emb).
    const int      shift     = 23 - m;
    const uint32_t round_bit = (mant_f >> (shift - 1)) & 1u;
    const uint32_t sticky    = (mant_f & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
    uint32_t       mant_t    = mant_f >> shift;
    mant_t += round_bit & (sticky | (mant_t & 1u));
    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) { mant_t = 0u; exp_out += 1; }
    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) { exp_out = max_exp; mant_t = (1u << m) - 1u; }
    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper C4: encode_fp8_emb_rhup_noflush. Round-half-up + proper subnormals.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint8_t encode_fp8_emb_rhup_noflush(float y, int e, int m, int b) {
    const uint32_t u    = __float_as_uint(y);
    const uint32_t sign = (u >> 31) & 1u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag == 0u) return 0u;

    const int32_t  exp_f  = int32_t((mag >> 23) & 0xFFu) - 127;
    const uint32_t mant_f = mag & 0x7FFFFFu;
    const int32_t  exp_t  = exp_f + b;

    if (exp_t < 1) {
        const int sub_shift = (23 - m) + (1 - exp_t);
        if (sub_shift > 24) return uint8_t(sign << 7);
        const uint32_t full_mant = (1u << 23) | mant_f;
        const uint32_t round_bit = (full_mant >> (sub_shift - 1)) & 1u;
        uint32_t mant_t = full_mant >> sub_shift;
        mant_t += round_bit;                               // round-half-up
        if (mant_t == (1u << m))
            return uint8_t((sign << 7) | (1u << m));       // overflow to smallest normal
        return uint8_t((sign << 7) | mant_t);
    }

    // Normal range (identical to encode_fp8_emb_rhup).
    const int      shift     = 23 - m;
    const uint32_t round_bit = (mant_f >> (shift - 1)) & 1u;
    uint32_t       mant_t    = mant_f >> shift;
    mant_t += round_bit;
    int32_t exp_out = exp_t;
    if (mant_t == (1u << m)) { mant_t = 0u; exp_out += 1; }
    const int32_t max_exp = (1 << e) - 1;
    if (exp_out > max_exp) { exp_out = max_exp; mant_t = (1u << m) - 1u; }
    return uint8_t((sign << 7) | (uint32_t(exp_out) << m) | mant_t);
}


// ----------------------------------------------------------------------------
// Helper D: decode_fp8_emb. FP8(e, m, b) -> FP32. Inverse of encode_fp8_emb.
// ----------------------------------------------------------------------------
__device__ __forceinline__
float decode_fp8_emb(uint8_t byte, int e, int m, int b) {
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
