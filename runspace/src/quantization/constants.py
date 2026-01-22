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
        return 15
    elif q_type == 'fp8_e4m3':
        return 7
    elif q_type == 'fp8_e3m4':
        return 3
    elif q_type == 'fp8_e2m5':
        return 1
    elif q_type == 'fp8_e1m6':
        return 0
    elif q_type == 'fp8_e6m1':
        return 31
    elif q_type == 'fp8_e7m0':
        return 63
    elif q_type == 'fp8_e0m7':
        return 0
    elif q_type == 'fp4_e2m1':
        return 2
    elif q_type == 'fp4_e3m0':
        return 4
    elif q_type == 'int8':
        return 7
    elif q_type == 'int4':
        return 3
    else:
        return 7


# Legacy constants for backward compatibility
FP8_E4M3_EXP_BIAS = 7
FP8_E5M2_EXP_BIAS = 15

FP32_MANT_BITS = 23
FP8_E4M3_MANT_BITS = 3
FP8_E5M2_MANT_BITS = 2
FP4_E2M1_MANT_BITS = 1
FP4_E3M0_MANT_BITS = 0
FP8_E0M7_MANT_BITS = 7

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
MANT_SHIFT_E0M7 = FP32_MANT_BITS - 7
