import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import os
from src.ops.quant_conv import QuantConv2d
from src.registry.op_registry import OpRegistry

class LayerComparator:
    def __init__(self, ref_model, quant_model, model_name="model", quant_type="quant", adapter=None, device=None, output_dir=None):
        self.ref_model = ref_model
        self.quant_model = quant_model
        self.model_name = model_name
        self.quant_type = quant_type
        self.adapter = adapter
        self.device = device if device else torch.device("cpu")
        self.output_dir = output_dir
        self.ref_activations = {}
        self.ref_weights = {}
        self.hooks = []
        
        # Metrics Engines
        from src.eval.metrics import MetricsEngine
        self.ref_metrics = MetricsEngine()
        self.quant_metrics = MetricsEngine()
        self.ref_certainty_sum = 0.0
        self.quant_certainty_sum = 0.0
        self.total_batches = 0
        self.layer_metrics = {}
        
        # Compression Stats
        from src.eval.compression import CompressionTracker
        self.compression_tracker = CompressionTracker()
        
        self._register_hooks()
        self._enable_quant_capture()
        self._load_compliance_config()

    def _load_compliance_config(self):
        """Loads compliance configuration from src/eval/compliance_config.yaml"""
        config_path = os.path.join(os.path.dirname(__file__), 'compliance_config.yaml')
        self.compliance_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.compliance_config = yaml.safe_load(f)
        else:
            print(f"Warning: Compliance config not found at {config_path}. Using defaults.")
            self.compliance_config = {
                'quantized_params': ['last_quant_rm', 'last_quant_rv', 'bias_quant', 'weight_fp8', 'weight']
            }

    def _register_hooks(self):
        def get_activation_hook(name):
            def hook(model, input):
                # Capture input to the layer
                # input is a tuple (x,)
                self.ref_activations[name] = input[0].detach().clone()
            return hook

        supported_ops = tuple(OpRegistry.get_supported_ops().keys())
        for name, module in self.ref_model.named_modules():
            if isinstance(module, supported_ops):
                self.hooks.append(module.register_forward_pre_hook(get_activation_hook(name)))
                if hasattr(module, 'weight') and module.weight is not None:
                    self.ref_weights[name] = module.weight.detach()

    def _enable_quant_capture(self):
        """
        Enables activation capture on the quantized model.
        Registers forward hooks on all leaf modules to capture inputs.
        """
        self.quant_activations = {}

        def get_hook(name):
            def hook(module, input, output):
                # Capture input (tuple) -> tensor
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input
                
                # Store detached clone to avoid memory issues or in-place modifications
                self.quant_activations[name] = inp.detach()
            return hook

        from src.ops.quant_base import QuantizedLayerMixin
        from src.ops.quant_mha import DecomposedMultiheadAttention

        quantized_ops = tuple(OpRegistry.get_supported_ops().values())

        for name, module in self.quant_model.named_modules():
            # Hook leaf modules (modules with no children)
            # For DecomposedMHA, it has children, but we want to capture its output too
            if len(list(module.children())) == 0 or isinstance(module, DecomposedMultiheadAttention):
                module.register_forward_hook(get_hook(name))
                
                # Enable internal capture for Quant layers
                if isinstance(module, quantized_ops):
                    module.capture_activations = True

    def compare(self, data_loader, num_batches=1, global_metrics=None):
        from src.eval.metrics import compute_certainty
        
        self.global_metrics = global_metrics
        self.ref_model.eval()
        self.quant_model.eval()
        
        # Run FX Coverage Verification BEFORE comparison loop
        # This ensures that if FX tracing leaves Proxies in the modules,
        # they will be overwritten by real data during the comparison loop.
        self.coverage_report_lines, self.unquantized_supported_count = self._verify_coverage_fx()
        
        print(f"Comparing models on {num_batches} batches...")
        
        with torch.no_grad():
            pbar = tqdm(enumerate(data_loader), total=num_batches, desc="Comparing", unit="batch")
            for i, batch in pbar:
                if i >= num_batches:
                    break
                
                # Prepare inputs for Ref Model (always unquantized)
                if hasattr(self, 'adapter') and hasattr(self.adapter, 'prepare_batch'):
                    # Temporarily disable input quantization for Ref Model
                    original_quant_setting = getattr(self.adapter, 'input_quantization', False)
                    if hasattr(self.adapter, 'input_quantization'):
                        self.adapter.input_quantization = False
                    
                    ref_inputs, ref_targets = self.adapter.prepare_batch(batch)
                    
                    # Restore setting
                    if hasattr(self.adapter, 'input_quantization'):
                        self.adapter.input_quantization = original_quant_setting
                else:
                    ref_inputs, ref_targets = batch

                # Prepare inputs for Quant Model (respects config)
                if hasattr(self, 'adapter') and hasattr(self.adapter, 'prepare_batch'):
                    quant_inputs, quant_targets = self.adapter.prepare_batch(batch)
                else:
                    quant_inputs, quant_targets = batch
                
                ref_inputs = ref_inputs.to(self.device)
                # ref_targets = ref_targets.to(self.device) # Targets are same
                quant_inputs = quant_inputs.to(self.device)
                quant_targets = quant_targets.to(self.device)
                targets = quant_targets # Use quant targets (should be same)
                
                # Run Ref Model
                self.ref_activations.clear() # Clear previous batch
                ref_outputs = self.ref_model(ref_inputs)
                
                # Run Quant Model
                quant_outputs = self.quant_model(quant_inputs)
                
                # Update Metrics
                self.ref_metrics.update(ref_outputs, targets)
                self.quant_metrics.update(quant_outputs, targets)
                
                self.ref_certainty_sum += compute_certainty(ref_outputs)
                self.quant_certainty_sum += compute_certainty(quant_outputs)
                self.total_batches += 1
                
                # Compare Layers
                self._compute_layer_metrics()

        self._generate_report()

    def _verify_coverage_fx(self):
        """
        Runs FX-based coverage verification.
        Returns a list of report lines.
        """
        import torch.nn as nn
        import torch.nn.functional as F
        from src.ops.quant_base import QuantizedLayerMixin
        from src.ops.quant_mha import DecomposedMultiheadAttention
        
        report_lines = []
        report_lines.append("--- Quantization Coverage Verification ---")
        
        try:
            import torch.fx
            
            # Custom Tracer to treat Quantized Layers as leaves
            class CoverageTracer(torch.fx.Tracer):
                def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                    # Treat DecomposedMHA as non-leaf to see internal Softmax/Linear
                    if isinstance(m, DecomposedMultiheadAttention):
                        return False
                    
                    # Treat other Quantized Layers and Activations as leaves
                    quantized_ops = tuple(OpRegistry.get_supported_ops().values())
                    if isinstance(m, quantized_ops):
                        return True
                        
                    return super().is_leaf_module(m, module_qualified_name)

            # We don't need to disable capture here because the subsequent comparison loop
            # will overwrite any Proxies captured during this trace.
            
            tracer = CoverageTracer()
            graph = tracer.trace(self.quant_model)
            traced = torch.fx.GraphModule(self.quant_model, graph)
            
            quantized_nodes = []
            unquantized_supported_nodes = []
            unquantized_unsupported_nodes = []
            
            # Define unquantized targets to look for
            supported_modules = tuple(OpRegistry.get_supported_ops().keys())
            quantized_ops = tuple(OpRegistry.get_supported_ops().values())
            
            supported_functions = OpRegistry.get_supported_functions()
            
            for node in traced.graph.nodes:
                if node.op == 'call_module':
                    module = self.quant_model.get_submodule(node.target)
                    if isinstance(module, quantized_ops):
                        quantized_nodes.append(f"{node.name} ({module.__class__.__name__})")
                    elif isinstance(module, supported_modules):
                        unquantized_supported_nodes.append(f"{node.name} ({module.__class__.__name__})")
                    else:
                        # Unsupported module (e.g. Identity, or custom layer not in registry)
                        unquantized_unsupported_nodes.append(f"{node.name} ({module.__class__.__name__})")
                        
                elif node.op == 'call_function':
                    if node.target in supported_functions:
                        unquantized_supported_nodes.append(f"{node.name} ({node.target.__name__})")
                    else:
                         # We generally ignore other functions unless they are clearly layers (like add/mul)
                         # For now, let's stick to modules + supported functions to avoid noise
                         pass
            
            total_leaves = len(quantized_nodes) + len(unquantized_supported_nodes) + len(unquantized_unsupported_nodes)
            coverage_pct = (len(quantized_nodes) / total_leaves * 100) if total_leaves > 0 else 0
            
            report_lines.append(f"Method: Automated Graph Tracing (torch.fx)")
            report_lines.append(f"Coverage: {len(quantized_nodes)}/{total_leaves} ({coverage_pct:.1f}%)")
            
            if unquantized_supported_nodes:
                report_lines.append("Unquantized Supported Ops (Should be quantized):")
                for node_str in unquantized_supported_nodes[:20]:
                    report_lines.append(f"  - {node_str}")
                if len(unquantized_supported_nodes) > 20:
                    report_lines.append(f"  ... and {len(unquantized_supported_nodes) - 20} more.")
            
            if unquantized_unsupported_nodes:
                report_lines.append("Unquantized Unsupported Ops (Not in registry):")
                for node_str in unquantized_unsupported_nodes[:20]:
                    report_lines.append(f"  - {node_str}")
                if len(unquantized_unsupported_nodes) > 20:
                    report_lines.append(f"  ... and {len(unquantized_unsupported_nodes) - 20} more.")
                
            if not unquantized_supported_nodes and not unquantized_unsupported_nodes:
                 report_lines.append("All ops are quantized! ✅")
                
        except Exception as e:
            report_lines.append(f"Method: Module Iteration (Fallback) - FX Trace Failed: {e}")
            # Fallback to existing module-based logic
            supported_layers = OpRegistry.get_supported_ops() # dict: original -> quantized
            quantized_ops = tuple(OpRegistry.get_supported_ops().values())
            
            quantized_count = 0
            unquantized_supported = []
            unquantized_unsupported = []
            
            # Iterate over all modules to find leaves
            for name, module in self.quant_model.named_modules():
                # Check if leaf (no children) OR DecomposedMHA
                if len(list(module.children())) == 0 or isinstance(module, DecomposedMultiheadAttention):
                    
                    if isinstance(module, QuantizedLayerMixin) or isinstance(module, quantized_ops):
                        quantized_count += 1
                    elif isinstance(module, tuple(supported_layers.keys())):
                        unquantized_supported.append(f"{name} ({module.__class__.__name__})")
                    else:
                        unquantized_unsupported.append(f"{name} ({module.__class__.__name__})")

            total_count = quantized_count + len(unquantized_supported) + len(unquantized_unsupported)
            coverage_pct = (quantized_count / total_count * 100) if total_count > 0 else 0
            
            report_lines.append(f"Coverage: {quantized_count}/{total_count} ({coverage_pct:.1f}%)")
            
            if unquantized_supported:
                report_lines.append("Unquantized Supported Layers:")
                for layer in unquantized_supported[:20]:
                    report_lines.append(f"  - {layer}")
            
            if unquantized_unsupported:
                report_lines.append("Unquantized Unsupported Layers:")
                for layer in unquantized_unsupported[:20]:
                    report_lines.append(f"  - {layer}")
            
        report_lines.append("-" * 40)
        report_lines.append("\n")
        
        # Calculate unquantized supported count
        unquantized_supported_count = 0
        if 'unquantized_supported_nodes' in locals():
            unquantized_supported_count = len(unquantized_supported_nodes)
        elif 'unquantized_supported' in locals():
            unquantized_supported_count = len(unquantized_supported)
            
        return report_lines, unquantized_supported_count

    def _compute_layer_metrics(self):
        from src.eval.metrics import compute_mse, compute_cosine_similarity
        from src.ops.quant_base import QuantizedLayerMixin
        from src.eval.compression import calculate_compression_stats
        
        # Track previous module type for fusion heuristic
        prev_module_type = None
        
        # We need to iterate in order. named_modules() is depth-first, which is usually correct for sequential models.
        # For complex graphs, this heuristic might be imperfect but sufficient for standard backbones.
        
        for name, module in self.quant_model.named_modules():
            # Determine if we should track this layer
            should_track = False
            quantized_ops = tuple(OpRegistry.get_supported_ops().values())
            
            if isinstance(module, quantized_ops):
                should_track = True
            elif name in self.quant_activations:
                should_track = True
            
            # Only process leaf modules for the fusion heuristic
            is_leaf = len(list(module.children())) == 0
            if not is_leaf:
                continue

            if not should_track:
                # Even if not tracking metrics, we might need to update prev_module_type if it's a relevant layer
                # But usually we only care about tracked layers for the report.
                # Let's update prev_module_type anyway if it's a "structural" layer
                # Use registry to determine structural layers (original classes)
                structural_ops = tuple(OpRegistry.get_supported_ops().keys())
                if isinstance(module, structural_ops):
                     prev_module_type = module.__class__.__name__
                continue
                
            if name not in self.ref_activations:
                # Still update prev_module_type
                prev_module_type = module.__class__.__name__
                continue
            
            # Initialize metrics for this layer if not present
            if name not in self.layer_metrics:
                self.layer_metrics[name] = {
                    'input_mse_sum': 0.0,
                    'input_cossim_sum': 0.0,
                    'weight_mse_sum': 0.0,
                    'weight_cossim_sum': 0.0,
                    'xmax_orig_sum': 0.0,
                    'xmax_deq_sum': 0.0,
                    'xmax_err_sum': 0.0,
                    'zeros_pct_input_sum': 0.0,
                    'zeros_pct_weight_sum': 0.0,
                    'count': 0,
                    'weight_count': 0,
                    'xmax_count': 0,
                    'xmax_count': 0,
                    'type': module.__class__.__name__
                }
            
            metrics = self.layer_metrics[name]
            
            # Determine bit width from q_type
            # Default to 8 bits
            bit_width = 8
            q_type = getattr(module, 'q_type', 'fp8_e4m3')
            if 'fp4' in q_type:
                bit_width = 4
            elif 'int8' in q_type or 'fp8' in q_type:
                bit_width = 8
            
            # 1. Input Metrics
            ref_input = self.ref_activations[name]
            quant_input = None
            
            if hasattr(module, 'last_quant_input'):
                 quant_input = module.last_quant_input
            elif name in self.quant_activations:
                 quant_input = self.quant_activations[name]
            
            if quant_input is not None:
                metrics['input_mse_sum'] += compute_mse(ref_input, quant_input)
                metrics['input_cossim_sum'] += compute_cosine_similarity(ref_input, quant_input)
                
                # Calculate zeros percentage
                zeros_pct = (quant_input == 0).float().mean().item() * 100.0
                metrics['zeros_pct_input_sum'] += zeros_pct
                
                # Compression Stats (Input & Weight)
                self.compression_tracker.update(module, prev_module_type, quant_input, bit_width=bit_width)
                
                metrics['count'] += 1
            
            # 2. Weight Metrics
            ref_weight = self.ref_weights.get(name, None)
            if ref_weight is not None and hasattr(module, 'last_quant_weight'):
                 quant_weight = module.last_quant_weight
                 metrics['weight_mse_sum'] += compute_mse(ref_weight, quant_weight)
                 metrics['weight_cossim_sum'] += compute_cosine_similarity(ref_weight, quant_weight)
                 
                 weight_zeros_pct = (quant_weight == 0).float().mean().item() * 100.0
                 metrics['zeros_pct_weight_sum'] += weight_zeros_pct
                 
                 metrics['weight_count'] += 1

            
            # 3. XMax Metrics
            if hasattr(module, 'last_quant_input_max') and hasattr(module, 'last_quant_input_dequant_max'):
                 orig_max = module.last_quant_input_max
                 dequant_max = module.last_quant_input_dequant_max
                 
                 # Avoid div by zero
                 eps = 1e-9
                 error = torch.abs(orig_max - dequant_max) / (orig_max + eps)
                 mean_error_pct = error.item() * 100.0
                 
                 metrics['xmax_orig_sum'] += orig_max.item()
                 metrics['xmax_deq_sum'] += dequant_max.item()
                 metrics['xmax_err_sum'] += mean_error_pct
                 metrics['xmax_count'] += 1
            
            # Update prev_module_type for next iteration
            prev_module_type = module.__class__.__name__

    def _generate_report(self):
        from src.eval.metrics import compute_mse, compute_cosine_similarity, compute_min_max, check_fp8_compliance
        from src.quantization.quantizer import get_fp8_e4m3_table, get_fp8_e5m2_table
        from src.ops.quant_base import QuantizedLayerMixin
        from src.ops.quant_mha import DecomposedMultiheadAttention
        import os
        import torch.nn as nn
        
        report_lines = []
        
        # 1. Model-Level Comparison
        if hasattr(self, 'global_metrics') and self.global_metrics:
            ref_acc = {'acc1': self.global_metrics.get('ref_acc1', 0.0), 'acc5': self.global_metrics.get('ref_acc5', 0.0)}
            quant_acc = {'acc1': self.global_metrics.get('acc1', 0.0), 'acc5': self.global_metrics.get('acc5', 0.0)}
            ref_certainty = self.global_metrics.get('ref_certainty', 0.0)
            quant_certainty = self.global_metrics.get('certainty', 0.0)
            acc_source = "Global (Full Dataset)"
        else:
            ref_acc = self.ref_metrics.compute()
            quant_acc = self.quant_metrics.compute()
            ref_certainty = self.ref_certainty_sum / max(1, self.total_batches)
            quant_certainty = self.quant_certainty_sum / max(1, self.total_batches)
            acc_source = "Local (Comparison Batches)"
        
        report_lines.append("\n=== Model Comparison Report ===")
        report_lines.append(f"Accuracy Source: {acc_source}")
        report_lines.append(f"{'Metric':<20} | {'Reference (FP32)':<20} | {f'Quantized ({self.quant_type})':<20}")
        report_lines.append("-" * 66)
        report_lines.append(f"{'Top-1 Accuracy':<20} | {ref_acc['acc1']:.2f}%{'':<13} | {quant_acc['acc1']:.2f}%")
        report_lines.append(f"{'Top-5 Accuracy':<20} | {ref_acc['acc5']:.2f}%{'':<13} | {quant_acc['acc5']:.2f}%")
        report_lines.append(f"{'Avg Certainty':<20} | {ref_certainty:.4f}{'':<14} | {quant_certainty:.4f}")
        
        # Compression Stats
        # Compression Stats
        report_lines.extend(self.compression_tracker.get_report_lines())
            
        report_lines.append("-" * 66)
        report_lines.append("\n")
        
        # 2. FP8 Compliance Check
        GREEN = "\033[92m"
        RED = "\033[91m"
        ORANGE = "\033[33m"
        RESET = "\033[0m"

        def format_status(passed, count, width, examples=None):
            if passed:
                status = "✅ PASS"
                padded = f"{status:<{width}}"
                return f"{GREEN}{padded}{RESET}"
            else:
                status = f"❌ FAIL ({count})"
                if examples:
                    status += f" {examples}"
                padded = f"{status:<{width}}"
                return f"{RED}{padded}{RESET}"

        report_lines.append("--- FP8 Compliance Check (Value-Based) ---")
        # Header: Layer(40) | Type(20) | Shape(15) | Weight Check(20) | Input Check(20)
        report_lines.append(f"{'Layer':<40} | {'Type':<20} | {'Shape':<15} | {'Weight Check':<20} | {'Input Check':<20}")
        report_lines.append("-" * 125)
        
        # Import FP8/FP4/INT8 table getters and compliance checker
        from ..quantization.quantizer import (
            get_fp8_e4m3_table, get_fp8_e5m2_table, 
            get_fp4_e2m1_table, get_fp4_e3m0_table, get_int8_table,
            check_fp8_compliance
        )
        
        # Pre-fetch tables
        device = next(self.quant_model.parameters()).device
        valid_values_fp8_e4m3 = get_fp8_e4m3_table(device)
        valid_values_fp8_e5m2 = get_fp8_e5m2_table(device)
        valid_values_fp4_e2m1 = get_fp4_e2m1_table(device)
        valid_values_fp4_e3m0 = get_fp4_e3m0_table(device)
        valid_values_int8 = get_int8_table(device)
        
        # Determine global q_type from the model (check first quantized layer)
        global_q_type = 'fp8_e4m3'
        for module in self.quant_model.modules():
            if hasattr(module, 'q_type'):
                global_q_type = module.q_type
                break
        
        # Store detailed failures for separate table
        detailed_failures = []
        
        for name, module in self.quant_model.named_modules():
            # Check leaf modules or DecomposedMHA
            if len(list(module.children())) == 0 or isinstance(module, DecomposedMultiheadAttention):
                layer_type = module.__class__.__name__
                
                # Determine q_type for this module
                # Use module's q_type if available, otherwise use global default
                q_type = getattr(module, 'q_type', global_q_type)
                
                if q_type == 'fp8_e5m2':
                    valid_values = valid_values_fp8_e5m2
                elif q_type == 'fp4_e2m1':
                    valid_values = valid_values_fp4_e2m1
                elif q_type == 'fp4_e3m0':
                    valid_values = valid_values_fp4_e3m0
                elif q_type == 'int8':
                    valid_values = valid_values_int8
                else:
                    valid_values = valid_values_fp8_e4m3
                
                # Check for Under Construction Status
                is_under_construction = OpRegistry.is_under_construction(layer_type)
                under_construction_str = f"{RED}{'🚧 UNDER CONSTRUCTION':<20}{RESET}"

                # Check Parameters (Weights + Stats)
                weight_str = f"{'N/A':<20}"
                
                if is_under_construction:
                     weight_str = under_construction_str
                # 1. Weights (Main Table)
                elif hasattr(module, 'weight_fp8') and module.weight_fp8 is not None:
                    passed, inv_count, examples = check_fp8_compliance(module.weight_fp8.float(), valid_values)
                    weight_str = format_status(passed, inv_count, 20, examples)
                    if not passed:
                        detailed_failures.append((name, layer_type, 'weight_fp8', inv_count, examples))
                elif hasattr(module, 'weight') and module.weight is not None:
                    passed, inv_count, examples = check_fp8_compliance(module.weight, valid_values)
                    weight_str = format_status(passed, inv_count, 20, examples)
                    if not passed:
                        detailed_failures.append((name, layer_type, 'weight', inv_count, examples))
                    
                # 2. Other Quantized Stats (Detailed Table Only)
                # Load from config or use default list
                potential_stats = self.compliance_config.get('quantized_params', ['last_quant_rm', 'last_quant_rv', 'bias_quant'])
                
                # Filter out weight/weight_fp8 as they are handled in main table (or should we just loop all?)
                # The user wanted main table to be "Weight Check".
                # So we keep weight/weight_fp8 separate for the main table logic, 
                # and use potential_stats for the detailed table (excluding weight/weight_fp8 if they are in there).
                
                for stat_name in potential_stats:
                    if stat_name in ['weight', 'weight_fp8']:
                        continue
                        
                    if hasattr(module, stat_name) and getattr(module, stat_name) is not None:
                        passed, inv_count, examples = check_fp8_compliance(getattr(module, stat_name).float(), valid_values)
                        if not passed:
                             detailed_failures.append((name, layer_type, stat_name, inv_count, examples))
                
                # Check Inputs
                input_str = f"{'N/A (No Capture)':<20}"
                
                # Check for custom compliance status in registry
                custom_status = OpRegistry.get_compliance_status(module.__class__.__name__)
                
                if is_under_construction:
                    if custom_status:
                        # Append under construction to custom status
                        input_str = f"{ORANGE}{custom_status}{RESET} {RED}🚧 UNDER CONSTRUCTION{RESET}"
                    else:
                        input_str = under_construction_str
                elif custom_status:
                     # Just custom status (Orange)
                     input_str = f"{ORANGE}{custom_status:<20}{RESET}"
                # Prefer internal unscaled capture for Quant layers
                elif hasattr(module, 'last_quant_input_unscaled') and module.last_quant_input_unscaled is not None:
                    input_passed, i_inv_count, i_examples = check_fp8_compliance(module.last_quant_input_unscaled, valid_values)
                    input_str = format_status(input_passed, i_inv_count, 20, i_examples)
                # Check output for activation layers (they quantize output, not input)
                elif hasattr(module, 'last_quant_output_unscaled') and module.last_quant_output_unscaled is not None:
                    input_passed, i_inv_count, i_examples = check_fp8_compliance(module.last_quant_output_unscaled, valid_values)
                    input_str = format_status(input_passed, i_inv_count, 20, i_examples)
                # Fallback to hook capture
                elif name in self.quant_activations:
                    # We captured the raw input, which is likely FP32 in a simulated quantization setup.
                    # Checking it against FP8 will fail and is technically incorrect unless we expect
                    # the previous layer to output quantized values (which QuantConv2d does not, it outputs FP32).
                    # So we mark it as N/A (FP32).
                    input_str = f"{'N/A (FP32 Input)':<20}"
                
                # Get Shape
                shape_str = "N/A"
                if name in self.ref_activations:
                    shape_str = str(tuple(self.ref_activations[name].shape[1:]))
                else:
                    # Fallback to internal capture
                    if hasattr(module, 'last_quant_input') and module.last_quant_input is not None:
                         shape_str = str(tuple(module.last_quant_input.shape[1:]))
                    elif hasattr(module, 'last_quant_input_unscaled') and module.last_quant_input_unscaled is not None:
                         shape_str = str(tuple(module.last_quant_input_unscaled.shape[1:]))
                    elif hasattr(module, 'last_quant_output_unscaled') and module.last_quant_output_unscaled is not None:
                         shape_str = str(tuple(module.last_quant_output_unscaled.shape[1:]))

                report_lines.append(f"{name:<40} | {layer_type:<20} | {shape_str:<15} | {weight_str} | {input_str}")
        report_lines.append("-" * 125)
        report_lines.append("-" * 125)
        report_lines.append("\n")
        
        # Detailed Parameter Compliance Table
        if detailed_failures:
            report_lines.append("--- Detailed Parameter Compliance Failures ---")
            report_lines.append(f"{'Layer':<40} | {'Param Name':<20} | {'Invalid Count':<15} | {'Examples'}")
            report_lines.append("-" * 110)
            for name, layer_type, param_name, count, examples in detailed_failures:
                 report_lines.append(f"{name:<40} | {param_name:<20} | {str(count):<15} | {examples}")
            report_lines.append("-" * 110)
            report_lines.append("\n")

        # 3. Quantization Coverage Verification
        if hasattr(self, 'coverage_report_lines'):
            report_lines.extend(self.coverage_report_lines)
        else:
            report_lines.append("--- Quantization Coverage Verification ---")
            report_lines.append("Coverage verification failed or was not run.")
            report_lines.append("-" * 40)
            report_lines.append("\n")

        # 4. First Layer Input Dynamic Range
        report_lines.append("--- First Layer Input Dynamic Range ---")
        for name, module in self.quant_model.named_modules():
            if isinstance(module, QuantizedLayerMixin) and getattr(module, 'is_first_layer', False):
                if name in self.ref_activations:
                    ref_input = self.ref_activations[name]
                    min_val, max_val = compute_min_max(ref_input)
                    report_lines.append(f"Layer: {name}")
                    report_lines.append(f"Min: {min_val:.4f}")
                    report_lines.append(f"Max: {max_val:.4f}")
                else:
                    report_lines.append(f"Layer: {name} - Input not captured")
                break # Only first layer
        report_lines.append("-" * 40)
        report_lines.append("\n")

        # 4. Layer-Level Comparison (Average)
        report_lines.append(f"--- Layer Comparison (Average over {self.total_batches} Batches) ---")
        report_lines.append(f"{'Layer':<30} | {'Type':<20} | {'Input MSE':<10} | {'Input CosSim':<12} | {'Weight MSE':<10} | {'Weight CosSim':<12} | {'XMax Orig':<10} | {'XMax Deq':<10} | {'XMax Err %':<10} | {'Zeros % (I)':<12} | {'Zeros % (W)':<12}")
        report_lines.append("-" * 185)
        
        for name, metrics in self.layer_metrics.items():
            count = max(1, metrics['count'])
            weight_count = max(1, metrics['weight_count'])
            xmax_count = max(1, metrics['xmax_count'])
            
            input_mse = metrics['input_mse_sum'] / count
            input_cossim = metrics['input_cossim_sum'] / count
            zeros_pct_input = metrics['zeros_pct_input_sum'] / count
            
            weight_mse_str = "N/A"
            weight_cossim_str = "N/A"
            zeros_pct_weight_str = "N/A"
            
            if metrics['weight_count'] > 0:
                weight_mse = metrics['weight_mse_sum'] / weight_count
                weight_cossim = metrics['weight_cossim_sum'] / weight_count
                weight_mse_str = f"{weight_mse:.2e}"
                weight_cossim_str = f"{weight_cossim:.4f}"
                
                zeros_pct_weight = metrics['zeros_pct_weight_sum'] / weight_count
                zeros_pct_weight_str = f"{zeros_pct_weight:.2f}%"
            
            xmax_orig_str = "N/A"
            xmax_deq_str = "N/A"
            xmax_err_str = "N/A"

            if metrics['xmax_count'] > 0:
                 mean_orig = metrics['xmax_orig_sum'] / xmax_count
                 mean_deq = metrics['xmax_deq_sum'] / xmax_count
                 mean_error_pct = metrics['xmax_err_sum'] / xmax_count
                 
                 xmax_orig_str = f"{mean_orig:.2f}"
                 xmax_deq_str = f"{mean_deq:.2f}"
                 xmax_err_str = f"{mean_error_pct:.2f}%"

            layer_type = metrics['type']
            line = f"{name:<30} | {layer_type:<20} | {input_mse:.2e}   | {input_cossim:.4f}       | {weight_mse_str:<10}   | {weight_cossim_str:<12} | {xmax_orig_str:<10} | {xmax_deq_str:<10} | {xmax_err_str:<10} | {f'{zeros_pct_input:.2f}%':<12} | {zeros_pct_weight_str:<12}"
            report_lines.append(line)
        
        report_lines.append("-" * 185)
        
        # Print Report (with colors)
        report_str = "\n".join(report_lines)
        print(report_str)
        
        # Save Report (without colors)
        import re
        def strip_ansi(text):
            """Remove ANSI escape codes from text."""
            ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
            return ansi_pattern.sub('', text)
        
        if self.output_dir:
            report_dir = self.output_dir
        else:
            report_dir = f"reports/{self.model_name}/{self.quant_type}"
            
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "comparison_report.txt")
        
        with open(report_path, "w") as f:
            f.write(strip_ansi(report_str))
        print(f"Report saved to {report_path}")

    def close(self):
        for hook in self.hooks:
            hook.remove()
