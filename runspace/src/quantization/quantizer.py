"""
FP8 E4M3 Quantizer - Hardware-Friendly Implementation

This module implements FP8 quantization using simple exponent/mantissa operations
that can be easily translated to hardware (FPGA/ASIC).

NOTE: This implementation does NOT support Infinity or NaN values.
All overflow values are clamped to the maximum representable value.

FP8 E4M3 Format:
- 1 sign bit (S)
- 4 exponent bits (E), bias = 7
- 3 mantissa bits (M)
- Value = (-1)^S * (1.M) * 2^(E-7) for normal (E > 0)
- Value = (-1)^S * (0.M) * 2^(-6) for subnormal (E = 0)
- Max value: 448 (E=15, M=6)
- No Infinity or NaN representation

FP8 E5M2 Format:
- 1 sign bit (S)
- 5 exponent bits (E), bias = 15
- 2 mantissa bits (M)
- Value = (-1)^S * (1.M) * 2^(E-15) for normal (E > 0)
- Value = (-1)^S * (0.M) * 2^(-14) for subnormal (E = 0)
- Max value: 57344 (E=30, M=3)
- No Infinity or NaN representation

FP4 E2M1 Format (OCP Microscaling):
- 1 sign bit
- 2 exponent bits, bias = 1
- 1 mantissa bit
- Range: +/- 6
- No Infinity or NaN representation

FP4 E3M0 Format (OCP Microscaling):
- 1 sign bit
- 3 exponent bits, bias = 3
- 0 mantissa bits
- Range: +/- 12
- No Infinity or NaN representation

INT8 Format:
- Symmetric signed 8-bit integer
- Range: [-128, 127] (256 values)
- Assumes input is pre-scaled to this range

INT4 Format:
- Symmetric signed 4-bit integer
- Range: [-8, 7] (16 values)
- Assumes input is pre-scaled to this range
"""

import torch

# ============================================================================
# FP8 E4M3 Lookup Table (All 256 possible values)
# ============================================================================

_FP8_TABLE_CACHE = {}


def _generate_fp8_e4m3_table(device: torch.device = None, bias: int = 7) -> torch.Tensor:
    """
    Generate lookup table of all 256 possible FP8 E4M3 float values.
    Used for assertion/validation that quantized values are valid FP8.
    
    NOTE: No NaN or Infinity values. Max value is at (exp=15, mant=6).
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_fp8_e4m3_bias{bias}"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(256):
        sign = (code >> 7) & 0x1
        exp = (code >> 3) & 0xF
        mant = code & 0x7
        
        # No NaN support - all values are normal or subnormal
        if exp == 0:
            value = (mant / 8.0) * (2.0 ** (1 - bias))
        else:
            value = (1.0 + mant / 8.0) * (2.0 ** (exp - bias))
        
        if sign:
            value = -value
        values.append(value)
    
    table = torch.tensor(values, dtype=torch.float32, device=device)
    _FP8_TABLE_CACHE[cache_key] = table
    return table


def get_fp8_e4m3_table(device: torch.device = None) -> torch.Tensor:
    """Returns the lookup table of all 256 FP8 E4M3 values."""
    return _generate_fp8_e4m3_table(device).clone()


# ============================================================================
# FP8 E5M2 Lookup Table
# ============================================================================

def _generate_fp8_e5m2_table(device: torch.device = None, bias: int = 15) -> torch.Tensor:
    """
    Generate lookup table of all 256 possible FP8 E5M2 float values.
    
    NOTE: No NaN or Infinity values. Max value is at (exp=30, mant=3).
    Values with exp=31 are treated as normal values (not special).
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_e5m2_bias{bias}"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(256):
        sign = (code >> 7) & 0x1
        exp = (code >> 2) & 0x1F
        mant = code & 0x3
        
        # No Inf/NaN support - all values are normal or subnormal
        if exp == 0:
            # Subnormal: (-1)^S * (0.M) * 2^(1-bias)
            value = (mant / 4.0) * (2.0 ** (1 - bias))
        else:
            # Normal: (-1)^S * (1.M) * 2^(E-bias)
            value = (1.0 + mant / 4.0) * (2.0 ** (exp - bias))
        
        if sign:
            value = -value
        values.append(value)
    
    table = torch.tensor(values, dtype=torch.float32, device=device)
    _FP8_TABLE_CACHE[cache_key] = table
    return table

def get_fp8_e5m2_table(device: torch.device = None) -> torch.Tensor:
    """Returns the lookup table of all 256 FP8 E5M2 values."""
    return _generate_fp8_e5m2_table(device).clone()


# ============================================================================
# FP4 Lookup Tables
# ============================================================================

def get_fp4_e2m1_table(device: torch.device = None) -> torch.Tensor:
    """Returns the lookup table of all 16 FP4 E2M1 values (bias=1)."""
    return _generate_fp_generic_table(device, total_bits=4, exp_bits=2, mant_bits=1, bias=1).clone()

def get_fp4_e3m0_table(device: torch.device = None) -> torch.Tensor:
    """Returns the lookup table of all 16 FP4 E3M0 values (bias=3)."""
    return _generate_fp_generic_table(device, total_bits=4, exp_bits=3, mant_bits=0, bias=3).clone()





# ============================================================================
# INT8 Lookup Table
# ============================================================================

def _generate_int8_table(device: torch.device = None) -> torch.Tensor:
    """
    Generate lookup table of all 256 possible INT8 values (casted to float).
    Range: [-128, 127]
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_int8"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    # Generate all int8 values
    values = torch.arange(-128, 128, dtype=torch.float32, device=device)
    
    _FP8_TABLE_CACHE[cache_key] = values
    return values

def get_int8_table(device: torch.device = None) -> torch.Tensor:
    return _generate_int8_table(device).clone()


# ============================================================================
# INT4 Lookup Table
# ============================================================================

def _generate_int4_table(device: torch.device = None) -> torch.Tensor:
    """
    Generate lookup table of all 16 possible INT4 values (casted to float).
    Range: [-8, 7]
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_int4"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    # Generate all int4 values
    values = torch.arange(-8, 8, dtype=torch.float32, device=device)
    
    _FP8_TABLE_CACHE[cache_key] = values
    return values

def get_int4_table(device: torch.device = None) -> torch.Tensor:
    return _generate_int4_table(device).clone()


# ============================================================================
# Constants
# ============================================================================

# Import bias constants from single source of truth


FP32_EXP_BIAS = 127
FP8_E4M3_EXP_BIAS = 7
FP8_E5M2_EXP_BIAS = 15

FP32_MANT_BITS = 23
FP8_E4M3_MANT_BITS = 3
FP8_E5M2_MANT_BITS = 2


# Shifts
MANT_SHIFT_E4M3 = FP32_MANT_BITS - 3
MANT_SHIFT_E5M2 = FP32_MANT_BITS - 2
MANT_SHIFT_E2M5 = FP32_MANT_BITS - 5
MANT_SHIFT_E3M4 = FP32_MANT_BITS - 4
MANT_SHIFT_E1M6 = FP32_MANT_BITS - 6
MANT_SHIFT_E6M1 = FP32_MANT_BITS - 1
MANT_SHIFT_E7M0 = FP32_MANT_BITS - 0



# ============================================================================
# Main Quantization Functions
# ============================================================================

def quantize(tensor: torch.Tensor, q_type: str = "fp8_e4m3", validate: bool = False, rounding: str = "nearest") -> torch.Tensor:
    """
    Quantize tensor to FP8 format.
    
    Args:
        tensor: Input FP32 tensor
        tensor: Input FP32 tensor
        q_type: Quantization type ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "fp4_e3m0")
        validate: If True, assert all values are valid FP8 (slow, for debugging)
        rounding: Rounding mode ("nearest" or "truncate")
        
    Returns:
        Quantized tensor with values constrained to valid FP8 values
    """
    # if q_type == "fp8_e4m3":
    #     # exp=15, mant=6 max value (no NaN support, clamp to max)
    #     result = quantize_fp_generic(tensor, exp_bits=4, mant_bits=3, bias=bias, rounding=rounding, clip_max_exp=15, clip_max_mant=6)
    #     if validate and (bias is None or bias == 7):
    #          assert_fp8_valid(result, q_type="fp8_e4m3", bias=bias)
    #     return result

    # elif q_type == "fp8_e5m2":
    #     # exp=30, mant=3 max value (clamp to max normal, no Inf)
    #     result = quantize_fp_generic(tensor, exp_bits=5, mant_bits=2, bias=bias, rounding=rounding, clip_max_exp=30, clip_max_mant=3)
    #     if validate:
    #          assert_fp8_valid(result, q_type="fp8_e5m2", bias=bias)
    #     return result
        
    if (q_type.startswith("fp") or q_type.startswith("ufp") or q_type.startswith("efp") or q_type.startswith("uefp")) and "_e" in q_type and "m" in q_type:
        # Generic handling for any fpX/ufpX/efpX/uefpX formats
        # Parse bits from string
        try:
            is_efp = "efp" in q_type
            if is_efp:
                is_signed = q_type.startswith("efp")
                prefix = "efp" if is_signed else "uefp"
            else:
                is_signed = q_type.startswith("fp")
                prefix = "fp" if is_signed else "ufp"
            
            # Expected format: [u]fp[bits]_e[exp]m[mant]
            fpx_part = q_type.split('_')[0] 
            e_part = q_type.split('_e')[1]
            exp_bits = int(e_part.split('m')[0])
            mant_bits = int(e_part.split('m')[1])
            total_bits_name = int(fpx_part.replace(prefix, ''))
            
            # Validation logic
            # For fp: 1 (S) + E + M
            # For ufp: 0 (S) + E + M
            calc_bits = (1 if is_signed else 0) + exp_bits + mant_bits
            
            # Warn if name doesn't match bits? (Optional)
                
        except:
             raise ValueError(f"Could not parse format {q_type}")

        # If unsigned, clamp to non-negative
        if not is_signed:
            # For unsigned, we only represent positive numbers. Negative inputs are clipped to 0.
            tensor = torch.relu(tensor)
             
        if is_efp:
            result = quantize_fp_generic(tensor, exp_bits, mant_bits, rounding=rounding, is_efp = True)
        else:
            result = quantize_fp_generic(tensor, exp_bits, mant_bits, rounding=rounding)
            # result_i32 = quantize_fp_generic_i32(tensor, exp_bits, mant_bits, rounding=rounding)
            
            # # result = torch.where(result == result_i32, 100, result)
            # print(q_type)
            
            # print(result)

            # print(result_i32)
            # exit()
        
        if validate:
             assert_fp8_valid(result, q_type=q_type)
        return result

    elif q_type == "int8":
        result = quantize_int8(tensor)
        if validate:
            assert_fp8_valid(result, q_type="int8")
        return result

    elif q_type == "int4":
        result = quantize_int4(tensor)
        if validate:
            assert_fp8_valid(result, q_type="int4")
        return result

    elif q_type == "tf32":
        return quantize_tf32(tensor)

    else:
        raise ValueError(f"Unsupported quantization type: {q_type}")


# ============================================================================
# Generic FP Table Generator
# ============================================================================

def _generate_fp_generic_table(device: torch.device = None, total_bits: int = 8, exp_bits: int = 4, mant_bits: int = 3, bias: int = 7, signed: bool = True) -> torch.Tensor:
    """
    Generate lookup table for any FP format (S + E + M).
    If signed=False, S=0.
    """
    if device is None:
        device = torch.device('cpu')
        
    cache_key = f"{device}_{'s' if signed else 'u'}fp{total_bits}_e{exp_bits}m{mant_bits}_bias{bias}"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
        
    num_codes = 1 << total_bits
    codes = torch.arange(num_codes, dtype=torch.int32, device=device)
    
    # Extract fields
    if signed:
        # Sign is MSB: [S, E...E, M...M]
        sign_shift = total_bits - 1
        exp_shift = mant_bits
        mant_mask = (1 << mant_bits) - 1
        exp_mask = (1 << exp_bits) - 1
        
        sign = (codes >> sign_shift) & 0x1
        exp = (codes >> exp_shift) & exp_mask
        mant = codes & mant_mask
    else:
        # Unsigned: [E...E, M...M]
        # Sign is implicit 0
        sign = torch.zeros_like(codes)
        exp_shift = mant_bits
        mant_mask = (1 << mant_bits) - 1
        exp_mask = (1 << exp_bits) - 1
        
        exp = (codes >> exp_shift) & exp_mask
        mant = codes & mant_mask
    
    values = _fp_generic_to_float_vectorized(sign, exp, mant, exp_bits, mant_bits, bias)
    
    _FP8_TABLE_CACHE[cache_key] = values
    return values

def _generate_efp_generic_table(device: torch.device = None, total_bits: int = 8, exp_bits: int = 4, mant_bits: int = 3, bias: int = 7, signed: bool = True) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
        
    cache_key = f"{device}_{'s' if signed else 'u'}efp{total_bits}_e{exp_bits}m{mant_bits}_bias{bias}"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
        
    num_codes = 1 << total_bits
    codes = torch.arange(num_codes, dtype=torch.int32, device=device)
    
    if signed:
        sign_shift = total_bits - 1
        sign = (codes >> sign_shift) & 0x1
        # Body is everything except sign
        body_mask = (1 << (total_bits - 1)) - 1
        body = codes & body_mask
    else:
        sign = torch.zeros_like(codes)
        body = codes
        
    # EFP logic (User Proposal):
    # 0 -> 0.
    # Positives 1..Max_Std -> Same.
    # Negatives -> Same (except -0).
    # -0 -> Max_Ext.
    
    # Check for Negative Zero code (Sign=1, Body=0 which is Exp=0, Mant=0)
    # Mask for Body (everything except sign)
    body_mask = (1 << (total_bits - 1)) - 1
    body = codes & body_mask
    
    is_neg_zero = (sign == 1) & (body == 0)
    
    # If Neg Zero -> Max_Ext code
    # Max_Std code for positive is body_mask in body part? No.
    # Max_Std code (pos) is just `max_std_code`.
    # Max_Ext code is `max_std_code + 1`.
    
    # We need to construct the Exp/Mant for the value we want.
    max_std_code = (1 << (exp_bits + mant_bits)) - 1
    limit_ext = max_std_code + 1
    
    ext_mant = limit_ext & ((1 << mant_bits) - 1)
    ext_exp = limit_ext >> mant_bits
    
    # Current Exp/Mant from codes
    # We can rely on `_fp_generic_to_float_vectorized` doing the right thing for standard codes.
    # But for Neg Zero, we want to inject `ext_exp` and `ext_mant` and Sign=0.
    
    # We need to extract exp/mant from `codes` first to have the "standard" values for non-overridden cases
    # (The existing implementation of this function didn't extract them before this block in the new version? 
    #  Wait, I need to check the full function context. 'exp' and 'mant' variables might be missing if I replaced too much or they were defined earlier.)
    
    # Re-extracting for safety as I might have broken the flow
    if signed:
        sign_shift = total_bits - 1
        exp_shift = mant_bits
        mant_mask = (1 << mant_bits) - 1
        exp_mask = (1 << exp_bits) - 1
        
        sign = (codes >> sign_shift) & 0x1
        exp = (codes >> exp_shift) & exp_mask
        mant = codes & mant_mask
    else:
        # Should not happen for EFP (signed)
        sign = torch.zeros_like(codes)
        exp = codes # placeholder
        mant = codes
        
    # Override
    final_sign = torch.where(is_neg_zero, torch.zeros_like(sign), sign)
    final_exp = torch.where(is_neg_zero, torch.full_like(exp, ext_exp), exp)
    final_mant = torch.where(is_neg_zero, torch.full_like(mant, ext_mant), mant)
    
    values = _fp_generic_to_float_vectorized(final_sign, final_exp, final_mant, exp_bits, mant_bits, bias)
    
    _FP8_TABLE_CACHE[cache_key] = values
    return values




def quantize_tf32(tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulate TF32 precision: 1 sign, 8 exponent, 10 mantissa.
    This is done by rounding FP32 (23 mantissa bits) to 10 mantissa bits.
    """
    # TF32 has 10 mantissa bits. FP32 has 23.
    # We need to mask out the lower 13 bits (23 - 10 = 13).
    # To round to nearest, we add 2^(13-1) = 2^12 before masking.
    
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    
    # Mask for exponent and sign (top 9 bits: 1 sign + 8 exp)
    # 0xFF800000 = 1111 1111 1000 ... (8 ones for exp, 1 for sign? No)
    # FP32: S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM
    #       1 8        23
    # TF32: S EEEEEEEE MMMMMMMMMM 0000000000000
    # Mask: 1 11111111 1111111111 0000000000000
    # Hex:  F  F       F          E   0   0   0 ? No.
    # 1 (sign) + 8 (exp) + 10 (mant) = 19 bits.
    # Remaining 13 bits are zero.
    # Mask = 0xFFFFE000
    # F = 1111
    # F = 1111
    # F = 1111
    # F = 1111 (16 bits) -> Wait.
    # 32 bits total.
    # Keep top 19 bits.
    # 0xFFF80000 ?
    # F=1111, F=1111, F=1111 (12), 8=1000 (13+3=16). No.
    # 19 bits = 16 + 3.
    # F F F (12) + E (1110) ? No, 3 bits is 7 (0111) or E (1110) depending on position.
    # Top 19 bits:
    # 1111 1111 1111 1111 1110 0000 ...
    # F    F    F    F    E    0 ...
    # 0xFFFFE000
    
    mask = 0xFFFFE000
    
    # Rounding: Add 2^12 (bit 12, 0-indexed)
    # 1 << 12 = 4096
    # But we need to be careful with sign and overflow.
    # A safer way for simulation is to just mask if we don't care about perfect rounding,
    # but for "hardware behavior" rounding is usually expected.
    
    # Simple truncation (round towards zero/minus infinity depending on sign representation):
    # return (i32 & mask).view(torch.float32).view(orig_shape)
    
    # Round to nearest:
    # We add 1<<(13-1) = 1<<12 to the integer representation, BUT only if it doesn't overflow the exponent.
    # This is tricky in integer domain.
    # Let's use a simpler approach:
    # 1. Cast to int
    # 2. Add rounding bias
    # 3. Mask
    
    # Note: This assumes positive numbers or 2's complement behavior which matches float representation order for positive floats.
    # For negative floats, adding to the integer representation might move it in the wrong direction if we treat it as signed int?
    # FP32 is sign-magnitude.
    # So we should operate on absolute value or handle sign.
    
    # Extract sign
    sign_mask = 0x80000000
    sign = i32 & sign_mask
    abs_i32 = i32 & ~sign_mask
    
    # Add rounding bias (1 << 12)
    # Check for NaN/Inf (exponent = 255)
    exp_mask = 0x7F800000
    is_nan_inf = (abs_i32 & exp_mask) == exp_mask
    
    # We only round normal/subnormal numbers
    # Add 0x1000 (1 << 12)
    rounded = abs_i32 + 0x1000
    
    # Check if rounding caused overflow into exponent bits that changes it to Inf?
    # Or if it overflowed mantissa completely.
    # If we overflow mantissa, exponent increments, which is correct for float rounding.
    
    # Apply mask
    masked = rounded & mask
    
    # Restore sign
    final_i32 = sign | masked
    
    # Restore NaN/Inf (don't touch them)
    final_i32 = torch.where(is_nan_inf, i32, final_i32)
    
    return final_i32.view(torch.float32).view(orig_shape)




def quantize_fp_generic(tensor: torch.Tensor, exp_bits: int, mant_bits: int, rounding: str = "nearest", clip_max_exp: int = None, clip_max_mant: int = None, is_efp: bool = False) -> torch.Tensor:
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)

    f32 = tensor_flat.view(torch.int32)
    sign = (f32 >> 31) & 0x1
    exp32 = (f32 >> 23) & 0xFF
    mant32 = f32 & 0x7FFFFF | (1 << 23)
    
    m_mask_number = 127 - 2**exp_bits + 2 - exp32

    m_mask_number = torch.where(m_mask_number < 0, 0, m_mask_number)
    
    residue =(mant32 >> (23 - (mant_bits + 1) + m_mask_number) & 0x1) << (23 - mant_bits + m_mask_number)

    mant_trunc = mant32 >> (23 - mant_bits + m_mask_number) << (23-mant_bits + m_mask_number)

    mant_trunc = mant_trunc + residue


    mant32 = mant_trunc & 0x7FFFFF


    overflow = mant_trunc >> 24 & 0x1
    underflow = mant_trunc >> 23 & 0x3

    exp32 = torch.where((overflow == 1), exp32 + 1, exp32)

    exp32 = torch.where((underflow == 0), 0, exp32)
    mant32 = torch.where((underflow == 0), 0, mant32)

    max_value_exp = 0x7F
    max_value_mant = (0xFFFF_FFFF << (23 - mant_bits)) & 0x7FFFFF

    if not is_efp:
        mant32 = torch.where((exp32 > max_value_exp), max_value_mant, mant32)
        exp32 = torch.where((exp32 > max_value_exp), max_value_exp, exp32)
    else:
        mant32 = torch.where((exp32 > max_value_exp) & (sign ==1), max_value_mant, mant32)
        exp32 = torch.where((exp32 > max_value_exp) & (sign ==1), max_value_exp, exp32)

    
    sign = sign << 31
    exp32 = exp32 << 23

    
    # mant32 = mant_trunc

    return (sign | exp32 | mant32).view(torch.float32).view(orig_shape)

    
        
    





    
    # # Subnormal handling
    # is_fp8_sub = fp8_exp < 1
    # shift = torch.where(is_fp8_sub, shift + (1 - fp8_exp), shift)
    # shift = torch.clamp(shift, max=31)

def quantize_fp_generic_i32(tensor: torch.Tensor, exp_bits: int, mant_bits: int, rounding: str = "nearest", clip_max_exp: int = None, clip_max_mant: int = None, is_efp: bool = False) -> torch.Tensor:
    """
    Generic FP quantization for any E/M split.
    Uses 'No Inf/NaN' policy (clamps to max representable).
    
    Args:
        tensor: Input tensor
        exp_bits: Number of exponent bits
        mant_bits: Number of mantissa bits
        rounding: Rounding mode
        clip_max_exp: Optional override for max allowable exponent (raw integer value)
        clip_max_mant: Optional override for max allowable mantissa at max exponent
    """
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    mant32 = i32 & 0x7FFFFF
    
    if exp_bits == 0:
        bias = 0
    elif exp_bits == 1:
        bias = 1
    else:
        bias = (1 << (exp_bits - 1)) - 1

    fp8_exp = exp32.int() - (127 - bias)
    
    mant_full = mant32 | 0x800000
    is_fp32_sub = (exp32 == 0)
    mant_full = torch.where(is_fp32_sub, mant32, mant_full)
    
    # Calculate shift
    # FP32 mantissa = 23 bits. FP8 mantissa = M bits.
    # Shift = 23 - M
    mant_shift_val = 23 - mant_bits
    shift = torch.full_like(fp8_exp, mant_shift_val)
    
    # Subnormal handling
    is_fp8_sub = fp8_exp < 1
    shift = torch.where(is_fp8_sub, shift + (1 - fp8_exp), shift)
    shift = torch.clamp(shift, max=31)
    
    if rounding == "nearest":
        round_bit = (1 << (shift - 1).clamp(min=0))
        mant_rounded = mant_full + round_bit
    else: # truncate
        mant_rounded = mant_full

    mant_shifted = mant_rounded >> shift
    
    # Overflow check
    overflow_threshold = 1 << (mant_bits + 1)
    is_efp_t = torch.tensor(is_efp, device=sign.device)  # scalar tensor, broadcastable
    cond = (sign == 1) | (~is_efp_t)                      # elementwise OR
    overflow = cond & (mant_shifted >= overflow_threshold)
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp8_exp = torch.where(overflow, fp8_exp + 1, fp8_exp)
    
    # Store
    implicit_one = 1 << mant_bits
    
    # Renormalize subnormal to normal if rounded up
    is_subnorm_to_norm = (fp8_exp == 0) & (mant_shifted >= implicit_one)
    fp8_exp = torch.where(is_subnorm_to_norm, torch.ones_like(fp8_exp), fp8_exp)

    # Special handling for Exp=0 (no normal numbers)
    if exp_bits > 0:
        is_normal = mant_shifted >= implicit_one
    else:
        is_normal = torch.zeros_like(mant_shifted, dtype=torch.bool)
    
    stored_exp = torch.where(is_normal, fp8_exp, torch.zeros_like(fp8_exp))
    mask = (1 << mant_bits) - 1
    stored_mant = torch.where(is_normal, mant_shifted & mask, mant_shifted)
    
    # Clamp to Max Value
    # Max Exp = 2^E - 1 (since we assume no NaN/Inf)
    if exp_bits > 0:
        if clip_max_exp is not None:
            max_exp_val = clip_max_exp
        else:
            max_exp_val = (1 << exp_bits) - 1
            
        if clip_max_mant is not None:
             max_mant_val = clip_max_mant
        else:
             max_mant_val = mask

        # If we overflowed beyond max representable
        # Logic: If exp > max_exp, clamp. 
        # OR if exp == max_exp AND mant > max_mant, clamp.
        is_overflow = (stored_exp > max_exp_val) | ((stored_exp == max_exp_val) & (stored_mant > max_mant_val))
        
        stored_exp = torch.where(is_overflow, torch.full_like(stored_exp, max_exp_val), stored_exp)
        stored_mant = torch.where(is_overflow, torch.full_like(stored_mant, max_mant_val), stored_mant)
    else:
        # For E=0, max value is just max mantissa (all 1s)
        max_exp_val = 0
        stored_exp = torch.zeros_like(stored_exp)
        
        # Check if we exceeded max mantissa bits? 
        if clip_max_mant is not None:
             limit_mant = clip_max_mant
        else:
             limit_mant = mask
             
        stored_mant = torch.clamp(stored_mant, max=limit_mant)
        
    
    # Handle NaN/Inf input (clamp to max)
    is_nan_inf = (exp32 == 255)
    
    # Use computed max values for clamping NaNs/Infs too
    # If exp_bits=0, max_exp_val is 0.
    # If exp_bits>0, it's what we computed above.
    stored_exp = torch.where(is_nan_inf, torch.full_like(stored_exp, max_exp_val), stored_exp)
    
    target_mant_clamp = max_mant_val if exp_bits > 0 else (clip_max_mant if clip_max_mant is not None else mask)
    stored_mant = torch.where(is_nan_inf, torch.full_like(stored_mant, target_mant_clamp), stored_mant)
    
    # Convert back to float
    result = _fp_generic_to_float_vectorized(sign, stored_exp, stored_mant, exp_bits, mant_bits, bias)
    return result.view(orig_shape)

def _fp_generic_to_float_vectorized(sign: torch.Tensor, exp: torch.Tensor, mant: torch.Tensor, exp_bits: int, mant_bits: int, bias: int) -> torch.Tensor:
    """Convert generic FP fields to float32."""
    if mant_bits > 0:
        div_factor = float(1 << mant_bits)
        m_val = mant.float() / div_factor
        m_val = torch.where(exp > 0, m_val + 1.0, m_val)
    else:
        # e.g. E7M0. Mantissa is 0. Normal implies 1.0. Subnormal implies 0.0.
        m_val = torch.where(exp > 0, torch.ones_like(exp, dtype=torch.float), torch.zeros_like(exp, dtype=torch.float))

    e_val = exp.float() - float(bias)
    e_val = torch.where(exp == 0, torch.full_like(e_val, 1.0 - float(bias)), e_val)
    
    result = m_val * torch.pow(2.0, e_val)
    result = torch.where(sign == 1, -result, result)
    return result





def quantize_efp_generic(tensor: torch.Tensor, exp_bits: int, mant_bits: int, rounding: str = "nearest") -> torch.Tensor:
    return quantize_fp_generic(tensor, exp_bits, mant_bits, rounding, is_efp=True)


def quantize_int8(tensor: torch.Tensor, rounding: str = "nearest") -> torch.Tensor:
    """
    INT8 quantization (symmetric).
    Assumes tensor is already scaled to [-127, 127] range.
    Rounds to nearest integer and clamps to [-127, 127].
    
    Args:
        tensor: Input tensor
        rounding: "nearest" or "truncate"
    """
    return quantize_int8_manual(tensor, rounding=rounding)

def quantize_int8_manual(tensor: torch.Tensor, rounding: str = "nearest") -> torch.Tensor:
    """
    Manual INT8 quantization using bitwise operations (simulating hardware).
    """
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    mant32 = i32 & 0x7FFFFF
    
    # 1.M * 2^(E-127)
    # We want Integer Value. 
    # Mantissa represents 1.M * 2^23 (if we treat 1.M as 1M...M)
    # Value = (Mant_Int * 2^-23) * 2^(E-127) = Mant_Int * 2^(E - 150)
    # To get integer: Shift Right by (150 - E)
    
    # Handle normal vs subnormal (exp=0)
    # Normals: Implicit 1 at bit 23
    mant_full = mant32 | 0x800000
    # Subnormals: No implicit 1. Effective exp is same as E=1 for shift purposes (-126)
    # Formula above for E=1: 1 - 150 = -149.
    # Subnormal val: 0.M * 2^-126 = M * 2^-23 * 2^-126 = M * 2^-149.
    # So for Exp=0, we use shift corresponding to E=1 (149), and no implicit 1.
    
    is_subnormal = (exp32 == 0)
    mant_full = torch.where(is_subnormal, mant32, mant_full)
    
    # Calculate shift
    # Shift = 150 - E
    # For subnormals (E=0), Shift = 149.
    eff_exp = torch.where(is_subnormal, torch.ones_like(exp32), exp32.int())
    shift = 150 - eff_exp
    
    # Clamp shift to 0 (for large inputs > 2^23, though they will be clamped anyway)
    shift = torch.clamp(shift, min=0, max=31) 
    
    if rounding == "nearest":
        # Add 0.5 (1 << (shift - 1))
        round_bit = (1 << (shift - 1).clamp(min=0))
        mant_rounded = mant_full + round_bit
    else:
        mant_rounded = mant_full
        
    int_val = mant_rounded >> shift
    
    # Apply sign
    # 2s complement negation: ~x + 1 or just -x since we are in signed int container
    int_val = torch.where(sign == 1, -int_val.float(), int_val.float())
    
    # Clamp to [-127, 127]
    result = torch.clamp(int_val, -127.0, 127.0)
    
    return result.view(orig_shape)


def quantize_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    INT4 quantization (symmetric).
    Assumes tensor is already scaled to [-7, 7] range.
    Rounds to nearest integer and clamps to [-8, 7].
    
    Note: Range is [-8, 7] (16 values) for symmetric 4-bit signed integer.
    """
    # Round to nearest integer
    result = torch.round(tensor)
    # Clamp to valid range [-8, 7]
    result = torch.clamp(result, -8.0, 7.0)
    return result







# ============================================================================
# Validation
# ============================================================================



def check_fp8_compliance(tensor: torch.Tensor, valid_table: torch.Tensor = None, rtol: float = 1e-5, q_type: str = "fp8_e4m3", bias: int = None) -> tuple:
    """
    Check if tensor values are compliant with the specified quantization type.
    Returns: (passed, invalid_count, examples)
    """
    # If valid_table is passed (legacy from comparator), we might use it or ignore it if we generate it inside.
    # The comparator passes valid_values as second arg.
    # But for INT8 we don't use table.
    # Let's handle table generation if not provided or if q_type requires specific handling.
    
    if q_type == "int8":
        # Memory-efficient check for INT8
        # 1. Check range
        if tensor.min() < -128 or tensor.max() > 127:
             # Find out of range
             mask = (tensor < -128) | (tensor > 127)
             count = mask.sum().item()
             examples = tensor[mask][:5].tolist()
             return False, count, examples
        
        # 2. Check if values are integers
        diff = torch.abs(tensor - torch.round(tensor))
        if diff.max() > rtol:
             mask = diff > rtol
             count = mask.sum().item()
             examples = tensor[mask][:5].tolist()
             return False, count, examples
        return True, 0, []

    if q_type == "int4":
        # Memory-efficient check for INT4
        # 1. Check range
        if tensor.min() < -8 or tensor.max() > 7:
             # Find out of range
             mask = (tensor < -8) | (tensor > 7)
             count = mask.sum().item()
             examples = tensor[mask][:5].tolist()
             return False, count, examples
        
        # 2. Check if values are integers
        diff = torch.abs(tensor - torch.round(tensor))
        if diff.max() > rtol:
             mask = diff > rtol
             count = mask.sum().item()
             examples = tensor[mask][:5].tolist()
             return False, count, examples
        return True, 0, []

    # For FP8/FP4/FP*, we need the table
    if valid_table is None:
        if q_type == "fp8_e4m3":
            valid_table = _generate_fp8_e4m3_table(tensor.device, bias=bias if bias is not None else 7)
        elif q_type == "fp8_e5m2":
            valid_table = _generate_fp8_e5m2_table(tensor.device, bias=bias if bias is not None else 15)
        elif (q_type.startswith("fp") or q_type.startswith("ufp") or q_type.startswith("efp") or q_type.startswith("uefp")) and "_e" in q_type and "m" in q_type:
             # Generic FP generation
             try:
                 is_efp = "efp" in q_type
                 if is_efp:
                     is_signed = q_type.startswith("efp")
                 else:
                     is_signed = q_type.startswith("fp")
                 
                 fpx_part = q_type.split('_')[0]
                 e_part = q_type.split('_e')[1]
                 exp_bits = int(e_part.split('m')[0])
                 mant_bits = int(e_part.split('m')[1])
                 
                 # Calculate stored bits
                 total_iter_bits = exp_bits + mant_bits
                 if is_signed:
                     total_iter_bits += 1
                     
                 # Calculate default bias if not provided
                 if bias is None:
                     bias = (1 << (exp_bits - 1)) - 1 if exp_bits > 0 else 0
                 
                 if is_efp:
                     valid_table = _generate_efp_generic_table(tensor.device, total_iter_bits, exp_bits, mant_bits, bias, signed=is_signed)
                 else:
                     valid_table = _generate_fp_generic_table(tensor.device, total_iter_bits, exp_bits, mant_bits, bias, signed=is_signed)
             except:
                  raise ValueError(f"Could not parse or generate table for format {q_type}")
        else:
             raise ValueError(f"Unknown q_type: {q_type}")

    valid_non_nan = valid_table[~torch.isnan(valid_table)]
    tensor_flat = tensor.contiguous().view(-1)
    non_nan_vals = tensor_flat[~torch.isnan(tensor_flat)]
    
    if len(non_nan_vals) == 0:
        return True, 0, []
        
    # Chunking to avoid OOM
    # With N=1M and M=256 (FP8), matrix is 1M * 256 * 4 bytes = 1GB.
    # This is fine for 1 config, but in parallel setup, multiple processes/threads might stress it?
    # No, this runs sequentially in main process usually.
    # However, for broader compatibility and speed, let's reduce chunk size purely for safety.
    chunk_size = 10000 
    num_chunks = (len(non_nan_vals) + chunk_size - 1) // chunk_size
    
    total_invalid = 0
    examples = []
    
    for i in range(num_chunks):
        chunk = non_nan_vals[i*chunk_size : (i+1)*chunk_size]
        
        # Broadcasting [Chunk, 1] - [1, Table] -> [Chunk, Table]
        # Memory: Chunk * TableSize * 4 bytes
        # 10k * 256 * 4 = 10MB. Very safe.
        diffs = torch.abs(chunk.unsqueeze(1) - valid_non_nan.unsqueeze(0))
        min_diffs = diffs.min(dim=1).values
        
        invalid_mask = min_diffs > rtol * torch.abs(chunk).clamp(min=1e-10)
        
        if invalid_mask.any():
            count = invalid_mask.sum().item()
            total_invalid += count
            if len(examples) < 5:
                examples.extend(chunk[invalid_mask][:5 - len(examples)].tolist())

    if total_invalid > 0:
        return False, total_invalid, examples
        
    return True, 0, []

def assert_fp8_valid(tensor: torch.Tensor, rtol: float = 1e-5, q_type: str = "fp8_e4m3", bias: int = None) -> None:
    """
    Assert all values are valid FP8 (vectorized check).
    Can be disabled via SKIP_FP8_VALIDATION env var.
    """
    import os
    if os.environ.get("SKIP_FP8_VALIDATION", "0") == "1":
        return

    passed, count, examples = check_fp8_compliance(tensor, rtol=rtol, q_type=q_type, bias=bias)
    if not passed:
        raise AssertionError(
            f"Found {count} invalid {q_type} values. "
            f"Examples: {examples}"
        )

def is_fp8_valid(tensor: torch.Tensor, rtol: float = 1e-5, q_type: str = "fp8_e4m3") -> bool:
    passed, _, _ = check_fp8_compliance(tensor, rtol=rtol, q_type=q_type)
    return passed


def get_q_type_bounds(q_type: str) -> float:
    """
    Returns the approximate maximum representable value (absolute) for a given q_type.
    Used for visualization scaling.
    """
    if q_type == "int8":
        return 128.0
    if q_type == "int4":
        return 8.0
    if q_type == "tf32":
        return 65536.0 # Arbitrary large number for TF32 (which is effectively unbounded in this context)

    # Standard presets overrides
    if q_type == "fp8_e4m3": return 512.0
    if q_type == "fp8_e5m2": return 58368.0
    
    # Generic Parsing
    try:
        # Expected: [u]fp[B]_e[E]m[M]
        # Ignore prefix
        e_part = q_type.split('_e')[1]
        exp_bits = int(e_part.split('m')[0])
        mant_bits = int(e_part.split('m')[1])
        
        # Calculate max value
        # Bias
        if exp_bits == 0:
            bias = 0
            # E0: Max = 0.Man...n * 2^(1-0)? No, usually fixed point.
            # Our implementation: 
            # Max Mantissa (11...1) = 1 - 2^-M
            # Value = MaxMant * 2^(1-Bias) ??
            # Without explicit bias: Bias = 0.
            # Value = (Mask / 2^M) * 2^1 ?
             
            # Let's check generic to float conversion:
            # m_val = mant / 2^M
            # exp=0 -> e_val = 1 - bias
            # Subnormal formula: 0.M * 2^(1-Bias)
            
            # Max mant = 2^M - 1
            # Val = ((2^M-1)/2^M) * 2^1 = (1 - 2^-M) * 2 = 2 - 2^(1-M).
            # Approx 2.0
            return 2.0
            
        else:
            bias = (1 << (exp_bits - 1)) - 1
            
            # Max Exp = 2^E - 1 (Assuming no NaN reservation for generic)
            max_exp_stored = (1 << exp_bits) - 1
            max_exp_val = max_exp_stored - bias
            
            # Max Mantissa = 1.11...1
            # Val = (1 + (2^M-1)/2^M) * 2^max_exp_val
            # Val = (2 - 2^-M) * 2^max_exp_val
            
            max_val = (2.0 - (1.0 / (1 << mant_bits))) * (2.0 ** max_exp_val)
            
            # Return slightly rounded up power of 2 for nice plotting
            import math
            log2_val = math.log2(max_val)
            return 2.0 ** math.ceil(log2_val)
            
    except Exception:
        # Fallback
        return 128.0


if __name__ == "__main__":
    tensor = torch.arange(-1.99, 1.99, 0.01)
    quantized = quantize(tensor, q_type="efp4_e3m0")
    # print(quantized)
    for x, q in zip(tensor, quantized):
        print(f"{x.item():.2f}  ->  {q.item():.4f}")