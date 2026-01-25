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
# FP4 E2M1 Lookup Table
# ============================================================================

def _generate_fp4_e2m1_table(device: torch.device = None, bias: int = 1) -> torch.Tensor:
    """
    Generate lookup table of all 16 possible FP4 E2M1 float values.
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_fp4_e2m1_bias{bias}"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(16):
        sign = (code >> 3) & 0x1
        exp = (code >> 1) & 0x3
        mant = code & 0x1
        
        if exp == 0:
            # Subnormal: (-1)^S * (0.M) * 2^(1-Bias)
            value = (mant / 2.0) * (2.0 ** (1 - bias))
        else:
            # Normal: (-1)^S * (1.M) * 2^(E-Bias)
            value = (1.0 + mant / 2.0) * (2.0 ** (exp - bias))
        
        if sign:
            value = -value
        values.append(value)
    
    table = torch.tensor(values, dtype=torch.float32, device=device)
    _FP8_TABLE_CACHE[cache_key] = table
    return table

def get_fp4_e2m1_table(device: torch.device = None) -> torch.Tensor:
    return _generate_fp4_e2m1_table(device).clone()


# ============================================================================
# FP4 E3M0 Lookup Table
# ============================================================================

def _generate_fp4_e3m0_table(device: torch.device = None, bias: int = 3) -> torch.Tensor:
    """
    Generate lookup table of all 16 possible FP4 E3M0 float values.
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_fp4_e3m0_bias{bias}"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(16):
        sign = (code >> 3) & 0x1
        exp = code & 0x7
        mant = 0 # No mantissa bits
        
        # Bias
        if exp == 0:
            # Subnormal? With 0 mantissa bits, 0.0 is 0.
            # (-1)^S * 0 * 2^(1-bias) = 0
            value = 0.0
        else:
            # Normal: (-1)^S * 1.0 * 2^(E-bias)
            value = 1.0 * (2.0 ** (exp - bias))
            
        if sign:
            value = -value
        values.append(value)
    
    table = torch.tensor(values, dtype=torch.float32, device=device)
    _FP8_TABLE_CACHE[cache_key] = table
    return table

def get_fp4_e3m0_table(device: torch.device = None) -> torch.Tensor:
    return _generate_fp4_e3m0_table(device).clone()


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
from .constants import get_quantization_bias

FP32_EXP_BIAS = 127
FP8_E4M3_EXP_BIAS = get_quantization_bias('fp8_e4m3')  # 7
FP8_E5M2_EXP_BIAS = get_quantization_bias('fp8_e5m2')  # 15

FP32_MANT_BITS = 23
FP8_E4M3_MANT_BITS = 3
FP8_E5M2_MANT_BITS = 2
FP4_E2M1_MANT_BITS = 1
FP4_E3M0_MANT_BITS = 0

# Shifts
MANT_SHIFT_E4M3 = FP32_MANT_BITS - 3
MANT_SHIFT_E5M2 = FP32_MANT_BITS - 2
MANT_SHIFT_E2M5 = FP32_MANT_BITS - 5
MANT_SHIFT_E3M4 = FP32_MANT_BITS - 4
MANT_SHIFT_E1M6 = FP32_MANT_BITS - 6
MANT_SHIFT_E6M1 = FP32_MANT_BITS - 1
MANT_SHIFT_E7M0 = FP32_MANT_BITS - 0
MANT_SHIFT_E2M1 = FP32_MANT_BITS - 1
MANT_SHIFT_E3M0 = FP32_MANT_BITS - 0


# ============================================================================
# Main Quantization Functions
# ============================================================================

def quantize(tensor: torch.Tensor, q_type: str = "fp8_e4m3", bias: int = None, validate: bool = True, rounding: str = "nearest") -> torch.Tensor:
    """
    Quantize tensor to FP8 format.
    
    Args:
        tensor: Input FP32 tensor
        q_type: Quantization type ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "fp4_e3m0")
        bias: Exponent bias (optional, overrides default)
        validate: If True, assert all values are valid FP8 (slow, for debugging)
        rounding: Rounding mode ("nearest" or "truncate")
        
    Returns:
        Quantized tensor with values constrained to valid FP8 values
    """
    if q_type == "fp8_e4m3":
        # exp=15, mant=6 max value (no NaN support, clamp to max)
        result = quantize_fp8_generic(tensor, exp_bits=4, mant_bits=3, bias=bias, rounding=rounding, clip_max_exp=15, clip_max_mant=6)
        if validate and (bias is None or bias == 7):
             assert_fp8_valid(result, q_type="fp8_e4m3", bias=bias)
        return result

    elif q_type == "fp8_e5m2":
        # exp=30, mant=3 max value (clamp to max normal, no Inf)
        result = quantize_fp8_generic(tensor, exp_bits=5, mant_bits=2, bias=bias, rounding=rounding, clip_max_exp=30, clip_max_mant=3)
        if validate:
             assert_fp8_valid(result, q_type="fp8_e5m2", bias=bias)
        return result
        
    elif (q_type.startswith("fp") or q_type.startswith("ufp")) and "_e" in q_type and "m" in q_type:
        # Generic handling for any fpX/ufpX formats
        # Parse bits from string
        try:
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
             
        result = quantize_fp8_generic(tensor, exp_bits, mant_bits, bias=bias, rounding=rounding)
        
        if validate:
             assert_fp8_valid(result, q_type=q_type, bias=bias)
        return result

    elif q_type == "fp4_e2m1":
        result = quantize_fp4_e2m1(tensor, bias=bias, rounding=rounding)
        if validate:
            assert_fp8_valid(result, q_type="fp4_e2m1", bias=bias)
        return result

    elif q_type == "fp4_e3m0":
        result = quantize_fp4_e3m0(tensor, bias=bias, rounding=rounding)
        if validate:
            assert_fp8_valid(result, q_type="fp4_e3m0", bias=bias)
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
    
    values = _fp8_generic_to_float_vectorized(sign, exp, mant, exp_bits, mant_bits, bias)
    
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





def quantize_fp8_generic(tensor: torch.Tensor, exp_bits: int, mant_bits: int, bias: int = None, rounding: str = "nearest", clip_max_exp: int = None, clip_max_mant: int = None) -> torch.Tensor:
    """
    Generic FP8 quantization for any E/M split.
    Uses 'No Inf/NaN' policy (clamps to max representable).
    
    Args:
        tensor: Input tensor
        exp_bits: Number of exponent bits
        mant_bits: Number of mantissa bits
        bias: Exponent bias (optional)
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
    
    if bias is None:
        # Default bias: 2^(E-1) - 1. For E=0, bias=0.
        if exp_bits == 0:
            bias = 0
        else:
            bias = (1 << (exp_bits - 1)) - 1

    # Re-bias exponent
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
    overflow = mant_shifted >= overflow_threshold
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp8_exp = torch.where(overflow, fp8_exp + 1, fp8_exp)
    
    # Store
    implicit_one = 1 << mant_bits
    
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
    result = _fp8_generic_to_float_vectorized(sign, stored_exp, stored_mant, exp_bits, mant_bits, bias)
    return result.view(orig_shape)

def _fp8_generic_to_float_vectorized(sign: torch.Tensor, exp: torch.Tensor, mant: torch.Tensor, exp_bits: int, mant_bits: int, bias: int) -> torch.Tensor:
    """Convert generic FP8 fields to float32."""
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


def quantize_fp4_e2m1(tensor: torch.Tensor, bias: int = None, rounding: str = "nearest") -> torch.Tensor:
    """
    FP4 E2M1 quantization.
    Bias: 1 (default)
    Mantissa: 1 bit
    
    NOTE: No inf/nan support. Overflow and inf/nan inputs are clamped to max value.
    """
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    mant32 = i32 & 0x7FFFFF
    
    if bias is None:
        bias = 1
        
    # Re-bias exponent: new_exp = old_exp - 127 + bias
    fp4_exp = exp32.int() - (127 - bias)
    
    mant_full = mant32 | 0x800000
    is_fp32_sub = (exp32 == 0)
    mant_full = torch.where(is_fp32_sub, mant32, mant_full)
    
    shift = torch.full_like(fp4_exp, MANT_SHIFT_E2M1)
    is_fp4_sub = fp4_exp < 1
    shift = torch.where(is_fp4_sub, shift + (1 - fp4_exp), shift)
    shift = torch.clamp(shift, max=31)
    
    if rounding == "nearest":
        round_bit = (1 << (shift - 1).clamp(min=0))
        mant_rounded = mant_full + round_bit
    else: # truncate
        mant_rounded = mant_full

    mant_shifted = mant_rounded >> shift
    
    # Overflow check (mantissa has 1 bit, implicit bit at pos 1, so >= 4 is overflow)
    overflow = mant_shifted >= 4
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp4_exp = torch.where(overflow, fp4_exp + 1, fp4_exp)
    
    is_normal = mant_shifted >= 2
    stored_exp = torch.where(is_normal, fp4_exp, torch.zeros_like(fp4_exp))
    stored_mant = torch.where(is_normal, mant_shifted & 0x1, mant_shifted)
    
    # Clamp to max exponent (3) and max mantissa (1)
    is_overflow = stored_exp > 3
    stored_exp = torch.where(is_overflow, torch.full_like(stored_exp, 3), stored_exp)
    stored_mant = torch.where(is_overflow, torch.full_like(stored_mant, 1), stored_mant)
    
    # Handle NaN/Inf: clamp to max value (exp=3, mant=1)
    is_nan_inf = (exp32 == 255)
    stored_exp = torch.where(is_nan_inf, torch.full_like(stored_exp, 3), stored_exp)
    stored_mant = torch.where(is_nan_inf, torch.full_like(stored_mant, 1), stored_mant)
    
    result = _fp4_e2m1_to_float_vectorized(sign, stored_exp, stored_mant, bias=bias)
    return result.view(orig_shape)


def quantize_fp4_e3m0(tensor: torch.Tensor, bias: int = None, rounding: str = "nearest") -> torch.Tensor:
    """
    FP4 E3M0 quantization.
    Bias: 3 (default)
    Mantissa: 0 bits
    
    NOTE: No inf/nan support. Overflow and inf/nan inputs are clamped to max value.
    """
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    mant32 = i32 & 0x7FFFFF
    
    if bias is None:
        bias = 3
        
    # Re-bias exponent
    fp4_exp = exp32.int() - (127 - bias)
    
    mant_full = mant32 | 0x800000
    is_fp32_sub = (exp32 == 0)
    mant_full = torch.where(is_fp32_sub, mant32, mant_full)
    
    shift = torch.full_like(fp4_exp, MANT_SHIFT_E3M0)
    is_fp4_sub = fp4_exp < 1
    shift = torch.where(is_fp4_sub, shift + (1 - fp4_exp), shift)
    shift = torch.clamp(shift, max=31)
    
    if rounding == "nearest":
        round_bit = (1 << (shift - 1).clamp(min=0))
        mant_rounded = mant_full + round_bit
    else: # truncate
        mant_rounded = mant_full

    mant_shifted = mant_rounded >> shift
    
    # Overflow check (mantissa has 0 bits, implicit bit at pos 0, so >= 2 is overflow)
    overflow = mant_shifted >= 2
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp4_exp = torch.where(overflow, fp4_exp + 1, fp4_exp)
    
    is_normal = mant_shifted >= 1
    stored_exp = torch.where(is_normal, fp4_exp, torch.zeros_like(fp4_exp))
    # stored_mant is always 0 for E3M0
    stored_mant = torch.zeros_like(stored_exp)
    
    # Clamp to max exponent (7)
    is_overflow = stored_exp > 7
    stored_exp = torch.where(is_overflow, torch.full_like(stored_exp, 7), stored_exp)
    
    # Handle NaN/Inf: clamp to max value (exp=7)
    is_nan_inf = (exp32 == 255)
    stored_exp = torch.where(is_nan_inf, torch.full_like(stored_exp, 7), stored_exp)
    
    result = _fp4_e3m0_to_float_vectorized(sign, stored_exp, stored_mant, bias=bias)
    return result.view(orig_shape)


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




def _fp4_e2m1_to_float_vectorized(sign: torch.Tensor, exp: torch.Tensor, mant: torch.Tensor, bias: int = 1) -> torch.Tensor:
    """Convert FP4 E2M1 fields to float32."""
    m_val = mant.float() / 2.0
    m_val = torch.where(exp > 0, m_val + 1.0, m_val)
    
    e_val = exp.float() - float(bias)
    # Subnormal handling: if exp=0, exponent is 1-bias
    e_val = torch.where(exp == 0, torch.full_like(e_val, 1.0 - float(bias)), e_val)
    
    result = m_val * torch.pow(2.0, e_val)
    result = torch.where(sign == 1, -result, result)
    return result

def _fp4_e3m0_to_float_vectorized(sign: torch.Tensor, exp: torch.Tensor, mant: torch.Tensor, bias: int = 3) -> torch.Tensor:
    """Convert FP4 E3M0 fields to float32."""
    # Mantissa is always 0 (implied 1.0 for normal, 0.0 for subnormal)
    m_val = torch.where(exp > 0, torch.ones_like(exp, dtype=torch.float), torch.zeros_like(exp, dtype=torch.float))
    
    e_val = exp.float() - float(bias)
    # Subnormal handling: if exp=0, exponent is 1-bias
    e_val = torch.where(exp == 0, torch.full_like(e_val, 1.0 - float(bias)), e_val)
    
    result = m_val * torch.pow(2.0, e_val)
    result = torch.where(sign == 1, -result, result)
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
        elif q_type == "fp4_e2m1":
            valid_table = _generate_fp4_e2m1_table(tensor.device, bias=bias if bias is not None else 1)
        elif q_type == "fp4_e3m0":
            valid_table = _generate_fp4_e3m0_table(tensor.device, bias=bias if bias is not None else 3)
        elif (q_type.startswith("fp") or q_type.startswith("ufp")) and "_e" in q_type and "m" in q_type:
             # Generic FP generation
             try:
                 is_signed = q_type.startswith("fp")
                 prefix = "fp" if is_signed else "ufp"
                 
                 fpx_part = q_type.split('_')[0]
                 e_part = q_type.split('_e')[1]
                 exp_bits = int(e_part.split('m')[0])
                 mant_bits = int(e_part.split('m')[1])
                 # total_bits = int(fpx_part.replace(prefix, '')) # trust calculation instead
                 
                 # Calculate stored bits
                 # For Signed: S + E + M
                 # For Unsigned: E + M
                 # NOTE: The table generator expects the number of bits to ITERATE over.
                 # If ufp8 implies 7 stored bits (E+M=7), then we iterate 2^7 values.
                 total_iter_bits = exp_bits + mant_bits
                 if is_signed:
                     total_iter_bits += 1
                     
                 # Calculate default bias if not provided
                 if bias is None:
                     bias = (1 << (exp_bits - 1)) - 1 if exp_bits > 0 else 0
                 
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
    chunk_size = 1024 * 1024 # 1M elements
    num_chunks = (len(non_nan_vals) + chunk_size - 1) // chunk_size
    
    total_invalid = 0
    examples = []
    
    for i in range(num_chunks):
        chunk = non_nan_vals[i*chunk_size : (i+1)*chunk_size]
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
    """
    passed, count, examples = check_fp8_compliance(tensor, rtol=rtol, q_type=q_type, bias=bias)
    if not passed:
        raise AssertionError(
            f"Found {count} invalid {q_type} values. "
            f"Examples: {examples}"
        )

def is_fp8_valid(tensor: torch.Tensor, rtol: float = 1e-5, q_type: str = "fp8_e4m3") -> bool:
    passed, _, _ = check_fp8_compliance(tensor, rtol=rtol, q_type=q_type)
    return passed

