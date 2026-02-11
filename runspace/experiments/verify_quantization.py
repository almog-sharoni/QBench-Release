import torch
import os
import json
import argparse
import sys
import math

def get_format_params(fmt_str):
    """
    Parses format string (e.g., 'fp4_e2m1') to get mantissa bits.
    """
    try:
        if fmt_str == 'fp32':
            return 23
        if '_e' in fmt_str and 'm' in fmt_str:
            # Format: fpX_eYmM
            parts = fmt_str.split('_')
            # Assuming last part is eYmM
            em_part = parts[-1] 
            if 'm' in em_part:
                mant_bits = int(em_part.split('m')[1])
                if 'e' in em_part:
                    exp_bits = int(em_part[1])
                    return mant_bits, exp_bits
    except:
        raise ValueError(f"Invalid format string: {fmt_str}")

def verify_mantissa(tensor, mant_bits, exp_bits):
    """
    Verifies that the tensor values have at most 'mant_bits' of information 
    in the FP32 mantissa.
    
    Logic: 
    1. Cast to int32 representation.
    2. Extract mantissa (lower 23 bits).
    3. Create a mask that keeps only the top 'mant_bits' of the 23-bit mantissa.
       Mask = 0x7FFFFF & ~((1 << (23 - mant_bits)) - 1)
    4. Check if (original_mantissa & mask) == original_mantissa.
    """
    if mant_bits >= 23:
        return True, 1.0
        
    # View as int32
    i32 = tensor.contiguous().view(torch.int32)
    
    # Extract mantissa (23 bits)
    mant_mask_full = 0x7FFFFF
    original_mantissa = i32 & mant_mask_full
    
    # Create mask for allowed bits
    # We want to KEEP the top 'mant_bits' and CLEAR the lower '23 - mant_bits'.
    # e.g. if mant_bits=1, we keep bit 22, clear 0-21.
    # Calculate x_max and exp_max
    x_max = tensor.abs().max()
    exp_max = (x_max.view(torch.int32) >> 23) & 0xFF
    
    # Calculate exponent of each element
    exp_tensor = (i32 >> 23) & 0xFF
    
    # Calculate delta
    delta = exp_max - exp_tensor
    # print("delta: ", delta) 
    
    # Adjust shift
    base_shift = 23 - mant_bits
    shift = base_shift + torch.max(delta - ((2**exp_bits) - 1), torch.zeros_like(delta))
    # print("shift: ", shift) 

    
    # Clamp shift to [0, 23] (if shift >= 23, all bits cleared)
    shift = torch.clamp(shift, min=0, max=23)

    allowed_mask = mant_mask_full & ~((1 << shift) - 1)
    # for item in allowed_mask:
    #     print("allowed mask: ", float_to_hex(item.item()))
    # exit(0)
    
    masked_mantissa = original_mantissa & allowed_mask
    
    # Check equality
    # We can compute a boolean accuracy
    matches = (original_mantissa == masked_mantissa)
    pass_rate = matches.float().mean().item()
    
    if pass_rate < 1.0:
        # Debug: Print first few failures
        failures = ~matches
        fail_indices = torch.nonzero(failures).squeeze()
        if fail_indices.dim() == 0: fail_indices = fail_indices.unsqueeze(0)
        
        print(f"    FAIL DEBUG (First 5):")
        for idx in fail_indices[:5]:
            val = tensor.flatten()[idx].item()
            orig_m = original_mantissa.flatten()[idx].item()
            masked_m = masked_mantissa.flatten()[idx].item()
            delta_val = delta.flatten()[idx].item()
            tensor_max_val = x_max.item()
            tensor_max_val_hex = float_to_hex(tensor_max_val)

            print(f"      Idx {idx}: Val={val}, Hex={float_to_hex(val)}, Mant={orig_m:06x} vs Masked={masked_m:06x}, Delta={delta_val}, Tensor Max={tensor_max_val}, Tensor Max Hex={tensor_max_val_hex}" , 'Mant bits: ', mant_bits, 'Exp bits: ', exp_bits) 
        
        # for idx, item in enumerate(tensor.flatten()):
        #     val = item.item()
        #     hex_val = float_to_hex(val)
        #     sign, exp, mant = hex_to_sign_exp_mant(hex_val)
        #     print(f"Val={val}, Hex={hex_val}, Mant={int_to_hex(mant)}, Exp={int_to_hex(exp)}, Sign={sign}")
    return pass_rate == 1.0, pass_rate

import struct
def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def int_to_hex(i):
    return hex(i)

def hex_to_sign_exp_mant(h):
    i32 = int(h, 16)
    sign = (i32 >> 31) & 1
    exp = (i32 >> 23) & 0xFF
    mant = i32 & 0x7FFFFF
    return sign, exp, mant

def main():
    parser = argparse.ArgumentParser(description="Verify Quantized Weights")
    parser.add_argument("--weights_file", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--map_file", type=str, required=True, help="Path to .json map file")
    parser.add_argument("--use_chunking", action="store_true", help="Expect chunked weights")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size (if chunking used)")
    
    args = parser.parse_args()
    
    print(f"Loading weights from {args.weights_file}...")
    try:
        state_dict = torch.load(args.weights_file, map_location='cpu')
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    print(f"Loading map from {args.map_file}...")
    try:
        with open(args.map_file, 'r') as f:
            quant_map = json.load(f)
    except Exception as e:
        print(f"Error loading map: {e}")
        return

    print("\n--- Starting Verification ---\n")
    
    all_passed = True
    
    # Iterate through map as it dictates what should be verified
    for layer_name, fmt_info in quant_map.items():
        weight_key = f"{layer_name}.weight"
        
        if weight_key not in state_dict:
            print(f"[WARN] Layer {layer_name} in map but weight not found in state dict.")
            continue
            
        weight = state_dict[weight_key]
        
        if args.use_chunking:
            if not isinstance(fmt_info, list):
                print(f"[FAIL] {layer_name}: Expected list of formats for chunking, got {type(fmt_info)}")
                all_passed = False
                continue
            
            # Replicate get_chunked_tensor logic
            if weight.dim() > 1:
                flat = weight.flatten(1)
                batch = weight.shape[0]
            else:
                flat = weight.flatten(0)
                batch = 1
                
            num_elements = flat.shape[-1]
            pad_len = 0
            if num_elements % args.chunk_size != 0:
                pad_len = args.chunk_size - (num_elements % args.chunk_size)
                flat = torch.nn.functional.pad(flat, (0, pad_len))
                
            num_chunks = flat.shape[-1] // args.chunk_size
            # [B, N, C]
            chunked = flat.view(batch, num_chunks, args.chunk_size)
            
            # Flatten to [B*N, C] to match the formats list (which is flat list of all chunks)
            chunked_flat = chunked.reshape(-1, args.chunk_size)
            
            formats = fmt_info
            
            if len(formats) != chunked_flat.shape[0]:
                print(f"[WARN] {layer_name}: Formats count {len(formats)} != Chunks count {chunked_flat.shape[0]}. Truncating/Padding check.")
                # If mismatch, check min
                n = min(len(formats), chunked_flat.shape[0])
            else:
                n = len(formats)
            
            layer_pass = True
            
            for i in range(n):
                chunk = chunked_flat[i]
                fmt = formats[i]
                
                mant_bits, exp_bits = get_format_params(fmt)
                if mant_bits is None:
                    # Generic or unknown, skip? or assumed full precision? 
                    # If it's fp32, mant_bits=23.
                    continue
                    
                passed, rate = verify_mantissa(chunk, mant_bits, exp_bits)
                # if not passed:
                #     print(f"[FAIL] {layer_name} chunk {i} ({fmt}): Pass Rate {rate:.2%}")
                #     layer_pass = False
                #     all_passed = False
                    
                    # Stop after a few failures per layer to avoid spam
                    # if i > 50: 
                    #     print("... (stopping errors for this layer)")
                    #     break
            
            # if layer_pass:
            #      print(f"[PASS] {layer_name}: Verified {n} chunks.")

        else:
            # Layer-wise
            fmt = fmt_info
            mant_bits, exp_bits = get_format_params(fmt)
            
            if mant_bits is None:
                print(f"[SKIP] {layer_name}: Unknown format {fmt}")
                continue
                
            passed, rate = verify_mantissa(weight, mant_bits, exp_bits)
            
            if passed:
                print(f"[PASS] {layer_name}: Format {fmt} (M={mant_bits})")
            else:
                print(f"[FAIL] {layer_name}: Format {fmt} (M={mant_bits}), Pass Rate {rate:.2%}")
                all_passed = False

    print("\n--- Verification Summary ---")
    if all_passed:
        print("SUCCESS: All weights match their specified quantization formats.")
    else:
        print("FAILURE: Some weights did not match their formats.")

if __name__ == "__main__":
    # pass_rate, _ = verify_mantissa(torch.tensor([1.75]), 3, 0)
    # print(pass_rate)
    main()
