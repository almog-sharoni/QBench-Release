"""
Quantization constants - Single source of truth for bias values.
"""

def get_quantization_bias(q_type: str) -> int:
    """
    Get the exponent bias for a given quantization type.
    
    This is the single source of truth for bias values used across
    the quantization framework.
    
    Args:
        q_type: Quantization type (fp8_e4m3, fp8_e5m2, fp4_e2m1, fp4_e3m0, int8, int4)
        
    Returns:
        Exponent bias value for the given type
    """
    if q_type == 'fp8_e5m2':
        return 15  # Standard FP8 E5M2 bias
    elif q_type == 'fp4_e2m1':
        return 2  # FP4 E2M1 bias
    elif q_type == 'fp4_e3m0':
        return 4  # FP4 E3M0 bias
    elif q_type == 'int8':
        return 7  # log2(127) ≈ 6.99, rounded to 7
    elif q_type == 'fp8_e4m3':
        return 7  # Standard FP8 E4M3 bias
    elif q_type == 'int4':
        return 3  # log2(7) ≈ 2.807, rounded to 3
    else:
        return 7  # Default fallback (FP8 E4M3)


# Legacy constants for backward compatibility
FP8_E4M3_EXP_BIAS = 7
FP8_E5M2_EXP_BIAS = 15
