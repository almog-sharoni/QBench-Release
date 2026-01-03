"""
FP8 E4M3 Quantizer - Hardware-Friendly Implementation

This module implements FP8 quantization using simple exponent/mantissa operations
that can be easily translated to hardware (FPGA/ASIC).

FP8 E4M3FN Format:
- 1 sign bit (S)
- 4 exponent bits (E), bias = 7
- 3 mantissa bits (M)
- Value = (-1)^S * (1.M) * 2^(E-7) for normal (E > 0)
- Value = (-1)^S * (0.M) * 2^(-6) for subnormal (E = 0)
- NaN: 0x7F and 0xFF (E=15, M=7)
- No Infinity representation
- Max value: 448 (0x7E)

FP4 E2M1 Format (OCP Microscaling):
- 1 sign bit
- 2 exponent bits, bias = 1
- 1 mantissa bit
- Range: +/- 6

FP4 E3M0 Format (OCP Microscaling):
- 1 sign bit
- 3 exponent bits, bias = 3
- 0 mantissa bits
- Range: +/- 12
"""

import torch

# ============================================================================
# FP8 E4M3 Lookup Table (All 256 possible values)
# ============================================================================

_FP8_TABLE_CACHE = {}


def _generate_fp8_e4m3_table(device: torch.device = None) -> torch.Tensor:
    """
    Generate lookup table of all 256 possible FP8 E4M3 float values.
    Used for assertion/validation that quantized values are valid FP8.
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = str(device)
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(256):
        sign = (code >> 7) & 0x1
        exp = (code >> 3) & 0xF
        mant = code & 0x7
        
        # NaN (exp=15, mant=7)
        if exp == 15 and mant == 7:
            values.append(float('nan'))
            continue
        
        if exp == 0:
            value = (mant / 8.0) * (2.0 ** -6)
        else:
            value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        
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

def _generate_fp8_e5m2_table(device: torch.device = None) -> torch.Tensor:
    """
    Generate lookup table of all 256 possible FP8 E5M2 float values.
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_e5m2"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(256):
        sign = (code >> 7) & 0x1
        exp = (code >> 2) & 0x1F
        mant = code & 0x3
        
        # NaN/Inf (exp=31)
        if exp == 31:
            if mant == 0:
                values.append(float('inf') if sign == 0 else float('-inf'))
            else:
                values.append(float('nan'))
            continue
        
        if exp == 0:
            # Subnormal: (-1)^S * (0.M) * 2^(-14)
            value = (mant / 4.0) * (2.0 ** -14)
        else:
            # Normal: (-1)^S * (1.M) * 2^(E-15)
            value = (1.0 + mant / 4.0) * (2.0 ** (exp - 15))
        
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

def _generate_fp4_e2m1_table(device: torch.device = None) -> torch.Tensor:
    """
    Generate lookup table of all 16 possible FP4 E2M1 float values.
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_fp4_e2m1"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(16):
        sign = (code >> 3) & 0x1
        exp = (code >> 1) & 0x3
        mant = code & 0x1
        
        if exp == 0:
            # Subnormal: (-1)^S * (0.M) * 2^(1-Bias) = (-1)^S * (0.M) * 2^0
            # Bias=1. 1-1=0.
            value = (mant / 2.0) * (2.0 ** 0)
        else:
            # Normal: (-1)^S * (1.M) * 2^(E-Bias)
            value = (1.0 + mant / 2.0) * (2.0 ** (exp - 1))
        
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

def _generate_fp4_e3m0_table(device: torch.device = None) -> torch.Tensor:
    """
    Generate lookup table of all 16 possible FP4 E3M0 float values.
    """
    if device is None:
        device = torch.device('cpu')
    
    cache_key = f"{device}_fp4_e3m0"
    if cache_key in _FP8_TABLE_CACHE:
        return _FP8_TABLE_CACHE[cache_key]
    
    values = []
    for code in range(16):
        sign = (code >> 3) & 0x1
        exp = code & 0x7
        mant = 0 # No mantissa bits
        
        # Bias = 3
        if exp == 0:
            # Subnormal? With 0 mantissa bits, 0.0 is 0.
            # (-1)^S * 0 * 2^(1-3) = 0
            value = 0.0
        else:
            # Normal: (-1)^S * 1.0 * 2^(E-3)
            value = 1.0 * (2.0 ** (exp - 3))
            
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
# Constants
# ============================================================================

FP32_EXP_BIAS = 127
FP8_E4M3_EXP_BIAS = 7
FP8_E5M2_EXP_BIAS = 15

FP32_MANT_BITS = 23
FP8_E4M3_MANT_BITS = 3
FP8_E5M2_MANT_BITS = 2
FP4_E2M1_MANT_BITS = 1
FP4_E3M0_MANT_BITS = 0

# Shifts
MANT_SHIFT_E4M3 = FP32_MANT_BITS - FP8_E4M3_MANT_BITS  # 20
MANT_SHIFT_E5M2 = FP32_MANT_BITS - FP8_E5M2_MANT_BITS  # 21
MANT_SHIFT_E2M1 = FP32_MANT_BITS - FP4_E2M1_MANT_BITS  # 22
MANT_SHIFT_E3M0 = FP32_MANT_BITS - FP4_E3M0_MANT_BITS  # 23


# ============================================================================
# Main Quantization Functions
# ============================================================================

def quantize(tensor: torch.Tensor, q_type: str = "fp8_e4m3", bias: int = None, validate: bool = True) -> torch.Tensor:
    """
    Quantize tensor to FP8 format.
    
    Args:
        tensor: Input FP32 tensor
        q_type: Quantization type ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "fp4_e3m0")
        bias: Exponent bias (optional, overrides default)
        validate: If True, assert all values are valid FP8 (slow, for debugging)
        
    Returns:
        Quantized tensor with values constrained to valid FP8 values
    """
    if q_type == "fp8_e4m3":
        # Use PyTorch native if available (fastest) AND if bias is standard (7)
        if hasattr(torch, 'float8_e4m3fn') and (bias is None or bias == 7):
            return tensor.to(torch.float8_e4m3fn).float()
        
        result = quantize_fp8_e4m3(tensor, bias=bias)
        if validate and (bias is None or bias == 7):
            assert_fp8_valid(result, q_type="fp8_e4m3")
        return result

    elif q_type == "fp8_e5m2":
        # Use PyTorch native if available
        if hasattr(torch, 'float8_e5m2'):
            return tensor.to(torch.float8_e5m2).float()
            
        result = quantize_fp8_e5m2(tensor)
        if validate:
            assert_fp8_valid(result, q_type="fp8_e5m2")
        return result

    elif q_type == "fp4_e2m1":
        result = quantize_fp4_e2m1(tensor, bias=bias)
        if validate:
            assert_fp8_valid(result, q_type="fp4_e2m1")
        return result

    elif q_type == "fp4_e3m0":
        result = quantize_fp4_e3m0(tensor, bias=bias)
        if validate:
            assert_fp8_valid(result, q_type="fp4_e3m0")
        return result

    elif q_type == "int8":
        result = quantize_int8(tensor)
        if validate:
            assert_fp8_valid(result, q_type="int8")
        return result

    else:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def quantize_fp8_e4m3(tensor: torch.Tensor, bias: int = None) -> torch.Tensor:
    """Hardware-friendly FP8 E4M3 quantization (fully vectorized)."""
    # ... existing implementation ...
    # For brevity, I'm assuming I don't need to re-paste the whole function if I can avoid it,
    # but replace_file_content requires exact match or full replacement.
    # Since I'm replacing the whole block from line 69 onwards, I need to include it.
    # Wait, the tool allows replacing chunks. I should have used multi_replace or just replaced the relevant parts.
    # But I need to insert new functions.
    # Let's re-implement quantize_fp8_e4m3 here to be safe and consistent.
    
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    mant32 = i32 & 0x7FFFFF
    
    if bias is None:
        bias = 7

    fp8_exp = exp32.int() - (127 - bias)
    mant_full = mant32 | 0x800000
    is_fp32_sub = (exp32 == 0)
    mant_full = torch.where(is_fp32_sub, mant32, mant_full)
    
    shift = torch.full_like(fp8_exp, MANT_SHIFT_E4M3)
    is_fp8_sub = fp8_exp < 1
    shift = torch.where(is_fp8_sub, shift + (1 - fp8_exp), shift)
    shift = torch.clamp(shift, max=31)
    
    round_bit = (1 << (shift - 1).clamp(min=0))
    mant_rounded = mant_full + round_bit
    mant_shifted = mant_rounded >> shift
    
    overflow = mant_shifted >= 16
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp8_exp = torch.where(overflow, fp8_exp + 1, fp8_exp)
    
    is_normal = mant_shifted >= 8
    stored_exp = torch.where(is_normal, fp8_exp, torch.zeros_like(fp8_exp))
    stored_mant = torch.where(is_normal, mant_shifted & 0x7, mant_shifted)
    
    is_overflow = (stored_exp > 15) | ((stored_exp == 15) & (stored_mant > 6))
    stored_exp = torch.where(is_overflow, torch.full_like(stored_exp, 15), stored_exp)
    stored_mant = torch.where(is_overflow, torch.full_like(stored_mant, 6), stored_mant)
    
    is_nan_inf = (exp32 == 255)
    stored_exp = torch.where(is_nan_inf, torch.full_like(stored_exp, 15), stored_exp)
    stored_mant = torch.where(is_nan_inf, torch.full_like(stored_mant, 7), stored_mant)
    
    result = _fp8_e4m3_to_float_vectorized(sign, stored_exp, stored_mant, bias=bias)
    return result.view(orig_shape)


def quantize_fp8_e5m2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Hardware-friendly FP8 E5M2 quantization.
    Bias: 15
    Mantissa: 2 bits
    """
    orig_shape = tensor.shape
    tensor_flat = tensor.contiguous().view(-1)
    
    i32 = tensor_flat.view(torch.int32)
    sign = (i32 >> 31) & 0x1
    exp32 = (i32 >> 23) & 0xFF
    mant32 = i32 & 0x7FFFFF
    
    # Re-bias exponent: new_exp = old_exp - 127 + 15
    fp8_exp = exp32.int() - 112
    
    mant_full = mant32 | 0x800000
    is_fp32_sub = (exp32 == 0)
    mant_full = torch.where(is_fp32_sub, mant32, mant_full)
    
    # Shift for 2 bits mantissa
    shift = torch.full_like(fp8_exp, MANT_SHIFT_E5M2)
    is_fp8_sub = fp8_exp < 1
    shift = torch.where(is_fp8_sub, shift + (1 - fp8_exp), shift)
    shift = torch.clamp(shift, max=31)
    
    round_bit = (1 << (shift - 1).clamp(min=0))
    mant_rounded = mant_full + round_bit
    mant_shifted = mant_rounded >> shift
    
    # Overflow check (mantissa has 2 bits, so >= 4 is overflow)
    overflow = mant_shifted >= 4
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp8_exp = torch.where(overflow, fp8_exp + 1, fp8_exp)
    
    is_normal = mant_shifted >= 2 # Implicit 1
    stored_exp = torch.where(is_normal, fp8_exp, torch.zeros_like(fp8_exp))
    stored_mant = torch.where(is_normal, mant_shifted & 0x3, mant_shifted)
    
    # Clamp to max exponent (31 is Inf/NaN, max normal is 30)
    # E5M2 has Inf support
    is_overflow = stored_exp >= 31
    stored_exp = torch.where(is_overflow, torch.full_like(stored_exp, 31), stored_exp)
    stored_mant = torch.where(is_overflow, torch.zeros_like(stored_mant), stored_mant) # Inf has 0 mantissa
    
    # Handle NaN/Inf input
    is_nan = (exp32 == 255) & (mant32 != 0)
    is_inf = (exp32 == 255) & (mant32 == 0)
    
    stored_exp = torch.where(is_nan, torch.full_like(stored_exp, 31), stored_exp)
    stored_mant = torch.where(is_nan, torch.full_like(stored_mant, 1), stored_mant) # NaN has non-zero mantissa
    
    stored_exp = torch.where(is_inf, torch.full_like(stored_exp, 31), stored_exp)
    stored_mant = torch.where(is_inf, torch.zeros_like(stored_mant), stored_mant)
    
    result = _fp8_e5m2_to_float_vectorized(sign, stored_exp, stored_mant)
    return result.view(orig_shape)


def quantize_fp4_e2m1(tensor: torch.Tensor, bias: int = None) -> torch.Tensor:
    """
    FP4 E2M1 quantization.
    Bias: 1 (default)
    Mantissa: 1 bit
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
    
    round_bit = (1 << (shift - 1).clamp(min=0))
    mant_rounded = mant_full + round_bit
    mant_shifted = mant_rounded >> shift
    
    # Overflow check (mantissa has 1 bit, implicit bit at pos 1, so >= 4 is overflow)
    overflow = mant_shifted >= 4
    mant_shifted = torch.where(overflow, mant_shifted >> 1, mant_shifted)
    fp4_exp = torch.where(overflow, fp4_exp + 1, fp4_exp)
    
    is_normal = mant_shifted >= 2
    stored_exp = torch.where(is_normal, fp4_exp, torch.zeros_like(fp4_exp))
    stored_mant = torch.where(is_normal, mant_shifted & 0x1, mant_shifted)
    
    # Clamp to max exponent (3)
    is_overflow = stored_exp > 3
    stored_exp = torch.where(is_overflow, torch.full_like(stored_exp, 3), stored_exp)
    stored_mant = torch.where(is_overflow, torch.full_like(stored_mant, 1), stored_mant)
    
    # Handle NaN/Inf
    is_nan_inf = (exp32 == 255)
    stored_exp = torch.where(is_nan_inf, torch.full_like(stored_exp, 3), stored_exp)
    stored_mant = torch.where(is_nan_inf, torch.full_like(stored_mant, 1), stored_mant)
    
    result = _fp4_e2m1_to_float_vectorized(sign, stored_exp, stored_mant, bias=bias)
    return result.view(orig_shape)


def quantize_fp4_e3m0(tensor: torch.Tensor, bias: int = None) -> torch.Tensor:
    """
    FP4 E3M0 quantization.
    Bias: 3 (default)
    Mantissa: 0 bits
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
    
    round_bit = (1 << (shift - 1).clamp(min=0))
    mant_rounded = mant_full + round_bit
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
    
    # Handle NaN/Inf
    is_nan_inf = (exp32 == 255)
    stored_exp = torch.where(is_nan_inf, torch.full_like(stored_exp, 7), stored_exp)
    
    result = _fp4_e3m0_to_float_vectorized(sign, stored_exp, stored_mant, bias=bias)
    return result.view(orig_shape)


def quantize_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    INT8 quantization (symmetric).
    Assumes tensor is already scaled to [-127, 127] range.
    Rounds to nearest integer and clamps to [-127, 127].
    """
    # Round to nearest integer
    result = torch.round(tensor)
    # Clamp to valid range [-127, 127]
    result = torch.clamp(result, -127.0, 127.0)
    return result


def _fp8_e4m3_to_float_vectorized(sign: torch.Tensor, exp: torch.Tensor, mant: torch.Tensor, bias: int = 7) -> torch.Tensor:
    """Convert FP8 E4M3 fields to float32."""
    is_nan = (exp == 15) & (mant == 7)
    
    m_val = mant.float() / 8.0
    m_val = torch.where(exp > 0, m_val + 1.0, m_val)
    
    e_val = exp.float() - float(bias)
    e_val = torch.where(exp == 0, torch.full_like(e_val, 1.0 - float(bias)), e_val)
    
    result = m_val * torch.pow(2.0, e_val)
    result = torch.where(sign == 1, -result, result)
    result = torch.where(is_nan, torch.full_like(result, float('nan')), result)
    return result

def _fp8_e5m2_to_float_vectorized(sign: torch.Tensor, exp: torch.Tensor, mant: torch.Tensor) -> torch.Tensor:
    """Convert FP8 E5M2 fields to float32."""
    is_nan = (exp == 31) & (mant != 0)
    is_inf = (exp == 31) & (mant == 0)
    
    m_val = mant.float() / 4.0
    m_val = torch.where(exp > 0, m_val + 1.0, m_val)
    
    e_val = exp.float() - 15.0
    e_val = torch.where(exp == 0, torch.full_like(e_val, -14.0), e_val)
    
    result = m_val * torch.pow(2.0, e_val)
    result = torch.where(sign == 1, -result, result)
    
    result = torch.where(is_inf, torch.full_like(result, float('inf')) * torch.where(sign==1, -1.0, 1.0), result)
    result = torch.where(is_nan, torch.full_like(result, float('nan')), result)
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

def check_fp8_compliance(tensor: torch.Tensor, valid_table: torch.Tensor = None, rtol: float = 1e-5, q_type: str = "fp8_e4m3") -> tuple:
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

    # For FP8/FP4, we need the table
    if valid_table is None:
        if q_type == "fp8_e4m3":
            valid_table = _generate_fp8_e4m3_table(tensor.device)
        elif q_type == "fp8_e5m2":
            valid_table = _generate_fp8_e5m2_table(tensor.device)
        elif q_type == "fp4_e2m1":
            valid_table = _generate_fp4_e2m1_table(tensor.device)
        elif q_type == "fp4_e3m0":
            valid_table = _generate_fp4_e3m0_table(tensor.device)
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


def assert_fp8_valid(tensor: torch.Tensor, rtol: float = 1e-5, q_type: str = "fp8_e4m3") -> None:
    """
    Assert all values are valid FP8 (vectorized check).
    """
    passed, count, examples = check_fp8_compliance(tensor, rtol=rtol, q_type=q_type)
    if not passed:
        raise AssertionError(
            f"Found {count} invalid {q_type} values. "
            f"Examples: {examples}"
        )

def is_fp8_valid(tensor: torch.Tensor, rtol: float = 1e-5, q_type: str = "fp8_e4m3") -> bool:
    passed, _, _ = check_fp8_compliance(tensor, rtol=rtol, q_type=q_type)
    return passed

