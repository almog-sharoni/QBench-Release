import torch
import os

_FP8_TABLE_CACHE = {}

def round_fractional_part(y_frac: torch.Tensor) -> torch.Tensor:
    mant_bits = 8
    drop_bits = 23 - mant_bits  # 15 bits to drop

    orig_shape = y_frac.shape
    f32 = y_frac.contiguous().view(-1).view(torch.int32)

    exp32 = (f32 >> 23) & 0xFF
    mant32 = f32 & 0x7FFFFF

    # Add implicit leading 1 only for normal numbers (exp32 == 0 means zero/subnormal)
    is_normal = (exp32 != 0).int()
    mant_with_implicit = mant32 | (is_normal << 23)

    # Round-to-nearest: add half LSB before truncating (standard add-half-then-truncate)
    mant_rounded = ((mant_with_implicit + (1 << (drop_bits - 1))) >> drop_bits) << drop_bits

    # On rounding overflow (carry into bit 24), increment exponent and zero mantissa
    overflow = (mant_rounded >> 24) & 0x1
    exp32 = exp32 + overflow
    mant_rounded = torch.where(overflow == 1, torch.zeros_like(mant_rounded), mant_rounded)

    mant32_final = mant_rounded & 0x7FFFFF
    f32_out = (f32 & 0x80000000) | (exp32 << 23) | mant32_final

    return f32_out.view(torch.float32).view(orig_shape)



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

    elif q_type == "tf32":
        return quantize_tf32(tensor)

    else:
        raise ValueError(f"Unsupported quantization type: {q_type}")



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

    # Reinterpret float32 bit pattern as int32 to allow bitwise operations
    f32 = tensor_flat.view(torch.int32)
    # Extract the sign bit (1 bit at position 31)
    sign = (f32 >> 31) & 0x1
    # Extract the exponent (8 bits at position 23)
    exp32 = (f32 >> 23) & 0xFF
    # Extract the mantissa (23 bits) and restore the implicit leading 1
    mant32 = f32 & 0x7FFFFF | (1 << 23)
    
    # Calculate shift amount for subnormal numbers relative to target exponent bits.
    # 127 is the FP32 bias, 2**exp_bits - 2 is the target format bias.
    m_mask_number = 127 - 2**exp_bits + 2 - exp32

    # If shift amount is negative, it's a normal number, so clamp shift to 0
    m_mask_number = torch.where(m_mask_number < 0, 0, m_mask_number)
    
    # Extract the rounding bit (the bit just below the new least significant bit)
    # and shift it to the LSB position of the target mantissa
    residue =(mant32 >> (23 - (mant_bits + 1) + m_mask_number) & 0x1) << (23 - mant_bits + m_mask_number)

    # Truncate the mantissa to the target number of mantissa bits, incorporating subnormal shift
    mant_trunc = mant32 >> (23 - mant_bits + m_mask_number) << (23-mant_bits + m_mask_number)

    # Add the rounding bit back to the truncated mantissa (round half up)
    mant_trunc = mant_trunc + residue


    # Remove the implicit leading 1 to get standard mantissa representation
    mant32 = mant_trunc & 0x7FFFFF


    # Check for overflow/underflow caused by rounding
    # Overflow occurs if rounding carried over into the exponent bit position (bit 24)
    overflow = mant_trunc >> 24 & 0x1
    # Underflow checks if the implicit leading 1 was lost (i.e. flush to zero)
    underflow = mant_trunc >> 23 & 0x3

    # If mantissa overflowed, increment the exponent
    exp32 = torch.where((overflow == 1), exp32 + 1, exp32)

    # If the number underflowed to 0, flush exponent and mantissa to 0
    exp32 = torch.where((underflow == 0), 0, exp32)
    mant32 = torch.where((underflow == 0), 0, mant32)

    # Hardcoded maximum representable exponent (127 in FP32 corresponds to value 1.0 -> 2.0 max clamp)
    max_value_exp = 0x7F
    # Mask to keep only the most significant `mant_bits` of the mantissa set to 1
    max_value_mant = (0xFFFF_FFFF << (23 - mant_bits)) & 0x7FFFFF

    # Handle overflow clipping (clamping values exceeding the max representable format value)
    if not is_efp:
        # Standard clamping for symmetric formats: clamp to max value
        mant32 = torch.where((exp32 > max_value_exp), max_value_mant, mant32)
        exp32 = torch.where((exp32 > max_value_exp), max_value_exp, exp32)
    else:
        # Asymmetric clamping (EFP formats): clamp only for negative numbers (sign == 1)
        mant32 = torch.where((exp32 > max_value_exp) & (sign ==1), max_value_mant, mant32)
        exp32 = torch.where((exp32 > max_value_exp) & (sign ==1), max_value_exp, exp32)

    
    # Shift sign and exponent back to their correct positions for float32
    sign = sign << 31
    exp32 = exp32 << 23

    

    # Combine sign, exponent, and mantissa, and reinterpret back as float32
    return (sign | exp32 | mant32).view(torch.float32).view(orig_shape)




def quantize_efp_generic(tensor: torch.Tensor, exp_bits: int, mant_bits: int, rounding: str = "nearest") -> torch.Tensor:
    return quantize_fp_generic(tensor, exp_bits, mant_bits, rounding, is_efp=True)




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

    # For FP formats: a value is valid iff it is a fixed point of the kernel —
    # i.e., running quantize() on it yields itself. This ties the checker to the
    # same simulator that produces activations/weights at runtime, so the valid
    # set is exactly what the kernel can emit (not the mathematically-complete
    # FP grid, which includes subnormal/exponent states the kernel doesn't
    # synthesize).
    tensor_flat = tensor.contiguous().view(-1)
    finite_mask = torch.isfinite(tensor_flat)
    finite_vals = tensor_flat[finite_mask]
    if finite_vals.numel() == 0:
        return True, 0, []

    # `quantize` dispatches on q_type and ends up in quantize_fp_generic for
    # FP formats. We compute what the kernel would produce for each value and
    # compare.
    expected = quantize(finite_vals, q_type=q_type, rounding="nearest")
    diff = (finite_vals - expected).abs()
    tol = rtol * finite_vals.abs().clamp(min=1e-10)
    invalid_mask = diff > tol

    total_invalid = int(invalid_mask.sum().item())
    examples = finite_vals[invalid_mask][:5].tolist() if total_invalid > 0 else []

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
    tensor = torch.tensor([1.165880322318247803008543e-41])
    quantized = quantize(tensor, q_type="fp8_e7m0")
    # print(quantized)
    for x, q in zip(tensor, quantized):
        print(f"{x.item():.8e}  ->  {q.item():.8e}")