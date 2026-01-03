import torch
from src.registry.op_registry import OpRegistry

def calculate_compression_stats(tensor: torch.Tensor, bit_width: int = 8):
    """
    Calculates compression statistics for a given tensor based on the scheme:
    - Zero: 1 bit
    - Non-Zero: (1 + bit_width) bits (1 prefix + data)
    
    Args:
        tensor: The input tensor (assumed to be quantized/integer-like values).
        bit_width: The bit width of the original data (e.g., 8 for FP8/INT8, 4 for FP4).
        
    Returns:
        tuple: (compressed_bits, original_bits)
    """
    numel = tensor.numel()
    original_bits = numel * bit_width
    
    zeros = (tensor == 0).sum().item()
    non_zeros = numel - zeros
    
    compressed_bits = (zeros * 1) + (non_zeros * (1 + bit_width))
    
    return compressed_bits, original_bits


class CompressionTracker:
    def __init__(self):
        self.total_input_compressed_bits = 0.0
        self.total_input_original_bits = 0.0
        self.total_input_candidates_bits = 0.0
        
        self.total_weight_compressed_bits = 0.0
        self.total_weight_original_bits = 0.0
        self.total_weight_candidates_bits = 0.0

    def update(self, module, prev_module_type, quant_input, bit_width=8):
        """
        Updates compression statistics for a single layer.
        """
        # 1. Input Compression
        if quant_input is not None:
            # Track total candidate bits for inputs
            self.total_input_candidates_bits += quant_input.numel() * bit_width
            
            # New Logic: Only compress inputs if the PREVIOUS layer was an activation.
            should_compress_input = False
            if prev_module_type and OpRegistry.is_activation(prev_module_type):
                should_compress_input = True
            
            if should_compress_input:
                c_bits, o_bits = calculate_compression_stats(quant_input, bit_width=bit_width)
                self.total_input_compressed_bits += c_bits
                self.total_input_original_bits += o_bits

        # 2. Weight Compression
        if hasattr(module, 'last_quant_weight'):
             quant_weight = module.last_quant_weight
             # Track total candidate bits for weights
             w_bits = quant_weight.numel() * bit_width
             self.total_weight_candidates_bits += w_bits
             
             c_bits, o_bits = calculate_compression_stats(quant_weight, bit_width=bit_width)
             self.total_weight_compressed_bits += c_bits
             self.total_weight_original_bits += o_bits

    def get_report_lines(self):
        """
        Generates report lines for compression statistics.
        """
        report_lines = []
        
        total_compressed = self.total_weight_compressed_bits + self.total_input_compressed_bits
        
        if self.total_weight_original_bits > 0:
            w_comp_ratio = self.total_weight_compressed_bits / self.total_weight_original_bits
            w_reduction_pct = (1.0 - w_comp_ratio) * 100.0
            
            w_share_pct = 0.0
            if total_compressed > 0:
                w_share_pct = (self.total_weight_compressed_bits / total_compressed) * 100.0
                
            report_lines.append(f"{'Weight Comp. Red. %':<20} | {'N/A':<20} | {w_reduction_pct:.2f}% ({w_share_pct:.1f}% of total compressed)")
        
        if self.total_input_original_bits > 0:
            i_comp_ratio = self.total_input_compressed_bits / self.total_input_original_bits
            i_reduction_pct = (1.0 - i_comp_ratio) * 100.0
            
            i_share_pct = 0.0
            if total_compressed > 0:
                i_share_pct = (self.total_input_compressed_bits / total_compressed) * 100.0
            
            report_lines.append(f"{'Input Comp. Red. %':<20} | {'N/A':<20} | {i_reduction_pct:.2f}% ({i_share_pct:.1f}% of total compressed)")
            
        return report_lines

    def get_stats(self):
        """
        Returns a dictionary of compression statistics.
        """
        stats = {
            'weight_compression_reduction': 0.0,
            'weight_share_of_total': 0.0,
            'input_compression_reduction': 0.0,
            'input_share_of_total': 0.0
        }
        
        total_compressed = self.total_weight_compressed_bits + self.total_input_compressed_bits
        
        if self.total_weight_original_bits > 0:
            w_comp_ratio = self.total_weight_compressed_bits / self.total_weight_original_bits
            stats['weight_compression_reduction'] = (1.0 - w_comp_ratio) * 100.0
            
            if total_compressed > 0:
                stats['weight_share_of_total'] = (self.total_weight_compressed_bits / total_compressed) * 100.0
        
        if self.total_input_original_bits > 0:
            i_comp_ratio = self.total_input_compressed_bits / self.total_input_original_bits
            stats['input_compression_reduction'] = (1.0 - i_comp_ratio) * 100.0
            
            if total_compressed > 0:
                stats['input_share_of_total'] = (self.total_input_compressed_bits / total_compressed) * 100.0
                
        return stats
