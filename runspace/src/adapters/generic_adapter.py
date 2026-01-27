import torch
import torch.nn as nn
import torchvision.models as models
from .base_adapter import BaseAdapter
from ..registry.op_registry import OpRegistry
try:
    import src.ops # Ensure all ops are registered
except ImportError:
    from .. import ops



class GenericAdapter(BaseAdapter):
    """
    A generic adapter that works with any PyTorch model from torchvision.
    Recursively replaces Conv2d, Linear, BatchNorm2d with quantized ops.
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        model: nn.Module = None,
        model_source: str = "torchvision",
        weights: str = None,
        input_quantization: bool = True,
        quantize_first_layer: bool = False,
        quantized_ops: list = None,
        excluded_ops: list = None,
        quantization_type: str = "fp8_e4m3",
        quantization_bias: int = None,
        layer_config: dict = None,
        per_chunk_format: bool = False,
        input_quantization_type: str = None,
        quant_mode: str = "tensor",
        chunk_size: int = None,
        weight_mode: str = "channel",
        weight_chunk_size: int = None,
        act_mode: str = "tensor",
        act_chunk_size: int = None,
        fold_layers: bool = False,
        simulate_tf32_accum: bool = False,
        rounding: str = "nearest",
        input_chunk_size: int = None,
        run_id: str = "default"
    ):
        super().__init__()
        self.model_name = model_name
        self.base_model_instance = model
        self.model_source = model_source
        self.weights = weights
        self.input_quantization = input_quantization
        self.quantize_first_layer = quantize_first_layer
        self.quantized_ops = quantized_ops if quantized_ops is not None else ["Conv2d"]
        self.excluded_ops = excluded_ops if excluded_ops is not None else []
        self.quantization_type = quantization_type
        self.quantization_bias = quantization_bias
        self.layer_config = layer_config if layer_config is not None else {}
        self.per_chunk_format = per_chunk_format
        self.input_quantization_type = input_quantization_type
        self.run_id = run_id
        

        
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.weight_mode = weight_mode
        self.weight_chunk_size = weight_chunk_size
        self.act_mode = act_mode
        self.act_chunk_size = act_chunk_size
        self.act_mode = act_mode
        self.act_chunk_size = act_chunk_size
        self.fold_layers = fold_layers
        self.simulate_tf32_accum = simulate_tf32_accum
        self.rounding = rounding
        self.input_chunk_size = input_chunk_size
        self.model = self.build_model(quantized=True)

    def _load_base_model(self) -> nn.Module:
        """Loads the base model from torchvision or other sources."""
        if self.base_model_instance is not None:
            # Return a copy to ensure we don't modify the original instance
            import copy
            return copy.deepcopy(self.base_model_instance)
            
        if self.model_source == "torchvision":
            return self._load_torchvision_model()
        else:
            raise ValueError(f"Unsupported model source: {self.model_source}")

    def _load_torchvision_model(self) -> nn.Module:
        """Load a model from torchvision.models."""
        if not hasattr(models, self.model_name):
            raise ValueError(
                f"Model '{self.model_name}' not found in torchvision.models. "
                f"Available models include: resnet18, resnet50, vgg16, mobilenet_v3_large, etc."
            )
        
        model_fn = getattr(models, self.model_name)
        
        # Handle weights parameter
        if self.weights:
            # Try to get weights enum (e.g., ResNet18_Weights.IMAGENET1K_V1)
            weights_enum = self._get_weights_enum()
            if weights_enum:
                return model_fn(weights=weights_enum)
            else:
                # Fall back to weights=True for legacy compatibility
                return model_fn(weights=True)
        else:
            return model_fn(weights=None)

    def _get_weights_enum(self):
        """Get the weights enum for the model if available."""
        # Build the weights class name (e.g., resnet18 -> ResNet18_Weights)
        # Handle common naming patterns
        model_lower = self.model_name.lower()
        
        # Try common weight class patterns
        weight_class_patterns = [
            f"{self.model_name.title().replace('_', '')}_Weights",  # resnet18 -> Resnet18_Weights
            f"{self.model_name.upper()}_Weights",  # vgg16 -> VGG16_Weights
        ]
        
        # Special cases for common models
        weight_class_mappings = {
            "resnet18": "ResNet18_Weights",
            "resnet34": "ResNet34_Weights",
            "resnet50": "ResNet50_Weights",
            "resnet101": "ResNet101_Weights",
            "resnet152": "ResNet152_Weights",
            "vgg11": "VGG11_Weights",
            "vgg13": "VGG13_Weights",
            "vgg16": "VGG16_Weights",
            "vgg19": "VGG19_Weights",
            "vgg11_bn": "VGG11_BN_Weights",
            "vgg13_bn": "VGG13_BN_Weights",
            "vgg16_bn": "VGG16_BN_Weights",
            "vgg19_bn": "VGG19_BN_Weights",
            "mobilenet_v2": "MobileNet_V2_Weights",
            "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
            "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
            "efficientnet_b0": "EfficientNet_B0_Weights",
            "efficientnet_b1": "EfficientNet_B1_Weights",
            "densenet121": "DenseNet121_Weights",
            "densenet169": "DenseNet169_Weights",
            "densenet201": "DenseNet201_Weights",
        }
        
        weight_class_name = weight_class_mappings.get(model_lower)
        
        if weight_class_name and hasattr(models, weight_class_name):
            weights_class = getattr(models, weight_class_name)
            # Try to get the specific weight (e.g., IMAGENET1K_V1)
            if hasattr(weights_class, self.weights):
                return getattr(weights_class, self.weights)
            # Fall back to DEFAULT
            elif hasattr(weights_class, "DEFAULT"):
                return weights_class.DEFAULT
        
        return None

    def _calibrate_model(self, model: nn.Module):
        """Calibrate quantized layers (pre-compute scales)."""
        for module in model.modules():
            if hasattr(module, 'calibrate_weights'):
                module.calibrate_weights()

    def _fold_layers(self, model: nn.Module):
        """
        Folds (fuses) layers like Conv+BN or Linear+BN.
        This is done before quantization to improve efficiency and accuracy.
        """
        import torch.quantization
        
        # Fusion requires eval mode
        model.eval()
        
        # Find pairs to fuse
        # We iterate over the model and look for Conv2d/Linear followed by BatchNorm
        # This is a heuristic that works for many standard models (ResNet, etc.)
        # For more complex graphs, a graph traversal (FX) would be better.
        
        modules_to_fuse = []
        
        # Helper to traverse and find sequences
        # We look at named_modules to find adjacent layers in the definition
        # This works for ResNet BasicBlock where conv is followed by bn
        
        # Strategy: Iterate over named_modules. Keep track of the "previous" module.
        # If prev is Conv and curr is BN, and they are "connected" (heuristic: same parent or sequential), fuse.
        
        # Better Strategy for standard models:
        # Iterate over all sub-modules. If a module is a Sequential or has a forward that executes sequentially,
        # we can inspect its children.
        
        # Let's try a global search on named_modules which is flattened.
        # We need to be careful about the names.
        # e.g. layer1.0.conv1, layer1.0.bn1
        
        named_modules = list(model.named_modules())
        
        i = 0
        while i < len(named_modules) - 1:
            name_curr, mod_curr = named_modules[i]
            name_next, mod_next = named_modules[i+1]
            
            # Check for Conv2d + BatchNorm2d
            if isinstance(mod_curr, nn.Conv2d) and isinstance(mod_next, nn.BatchNorm2d):
                # Check if they are likely connected
                # Heuristic: names share a prefix and are likely sequential
                # e.g. ...conv1 and ...bn1
                # or just trusting the order in named_modules for standard torchvision models
                
                # Verify they are in the same container or adjacent
                # For ResNet: layer1.0.conv1, layer1.0.bn1 -> fused
                
                # We simply add them to the list
                modules_to_fuse.append([name_curr, name_next])
                i += 2 # Skip next since we consumed it
                continue
                
            # Check for Linear + BatchNorm1d
            if isinstance(mod_curr, nn.Linear) and isinstance(mod_next, nn.BatchNorm1d):
                modules_to_fuse.append([name_curr, name_next])
                i += 2
                continue
                
            i += 1
            
        if modules_to_fuse:
            # print(f"Fusing layers: {modules_to_fuse}")
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

    def _replace_layers(self, model: nn.Module):
        """Replace standard layers with quantized versions."""
        self._first_layer_found = False
        self._recursive_replace(model)

    def _create_quantized_module(self, module: nn.Module, QuantClass: type, name: str = "") -> nn.Module:
        """Creates a quantized module from the original module."""
        from ..ops.quant_base import QuantizedLayerMixin
        
        # Determine quantization settings for this module
        q_type = self.quantization_type
        bias = self.quantization_bias
        input_q_type = self.input_quantization_type # Default to global input format
        
        # Default modes
        input_mode = self.quant_mode
        act_chunk_size = self.act_chunk_size
        rounding = self.rounding
        input_chunk_size = self.input_chunk_size if self.input_chunk_size is not None else self.chunk_size
        
        # Restore missing assignments
        weight_mode = self.weight_mode
        weight_chunk_size = self.weight_chunk_size
        act_mode = self.act_mode
        
        
        # Check for layer-specific config
        layer_conf = {}
        if name in self.layer_config:
            layer_conf = self.layer_config[name]
            # Support both 'type' and 'format' keys for q_type
            if 'type' in layer_conf:
                q_type = layer_conf['type']
            elif 'format' in layer_conf:
                q_type = layer_conf['format']
                
        if 'bias' in layer_conf:
            bias = layer_conf['bias']
            
        if 'input_format' in layer_conf:
            input_q_type = layer_conf['input_format']
            
        if 'mode' in layer_conf:
            input_mode = layer_conf['mode']
        if 'chunk_size' in layer_conf:
            input_chunk_size = layer_conf['chunk_size']
        if 'weight_mode' in layer_conf:
            weight_mode = layer_conf['weight_mode']
        if 'weight_chunk_size' in layer_conf:
            weight_chunk_size = layer_conf['weight_chunk_size']
        if 'act_mode' in layer_conf:
            act_mode = layer_conf['act_mode']
        if 'act_chunk_size' in layer_conf:
            act_chunk_size = layer_conf['act_chunk_size']
        if 'rounding' in layer_conf:
            rounding = layer_conf['rounding']
        
        # Case 1: Custom from_native factory (e.g. MHA, BasicConv2d)
        if hasattr(QuantClass, 'from_native'):
             return QuantClass.from_native(module, q_type=q_type, quantization_bias=bias)

        # Case 2: Layer with weights (Conv, Linear, BN) - In-place class swap
        # We check if the QuantClass is a subclass of QuantizedLayerMixin
        if issubclass(QuantClass, QuantizedLayerMixin):
            # Perform in-place class swap to preserve weights
            module.__class__ = QuantClass
            
            # Initialize Mixin state
            module.q_type = q_type
            module.quantization_bias = bias
            module.input_q_type = input_q_type if input_q_type else q_type
            
            module.input_quantization = self.input_quantization
            module.input_mode = input_mode
            module.input_chunk_size = input_chunk_size
            module.weight_mode = weight_mode
            module.weight_chunk_size = weight_chunk_size
            module.rounding = rounding
            
            # Per-chunk format configuration
            if self.per_chunk_format and 'chunk_formats' in layer_conf:
                module.chunk_formats = layer_conf['chunk_formats']
            
            module.register_buffer('weight_scale', None)
            module.register_buffer('weight_fp8', None)
            
            # Set TF32 simulation flag if supported (QuantConv2d)
            if QuantClass.__name__ == "QuantConv2d":
                module.simulate_tf32_accum = self.simulate_tf32_accum
            
            # Handle first layer logic
            if isinstance(module, nn.Conv2d):
                 is_first = not self._first_layer_found
                 if is_first:
                     self._first_layer_found = True
                 
                 module.is_first_layer = is_first
                 if module.in_channels == 3 and is_first:
                     module.quantize_first_layer = self.quantize_first_layer
            
            return module

        # Case 3: Activations / Softmax - New instance
        kwargs = {'q_type': q_type}
        if bias is not None:
            kwargs['quantization_bias'] = bias
            
        # Pass input mode params to activation (activations only have inputs)
        # Use act_mode if available, otherwise fallback to input_mode?
        # No, user explicitly asked for separate mode for activations.
        kwargs['quant_mode'] = act_mode
        kwargs['chunk_size'] = act_chunk_size
        
        # Copy common attributes if they exist
        if hasattr(module, 'inplace'):
            kwargs['inplace'] = module.inplace
        if hasattr(module, 'approximate'):
            kwargs['approximate'] = module.approximate
        if hasattr(module, 'dim'):
            kwargs['dim'] = module.dim
            
        return QuantClass(**kwargs)

    def _recursive_replace(self, model: nn.Module, prefix: str = ""):
        """Recursively traverse and replace layers using the registry."""
        # Check for "quantize all" mode
        quantize_all = "-1" in self.quantized_ops or "all" in self.quantized_ops
        
        # Get supported ops from registry
        supported_ops = OpRegistry.get_supported_ops()

        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            new_module = None
            should_quantize = False
            
            # Check if module type is supported
            if type(module) in supported_ops:
                # Check if this specific op type is requested
                # We use the class name (e.g., "Conv2d", "ReLU")
                op_name = module.__class__.__name__
                
                if quantize_all or op_name in self.quantized_ops:
                    should_quantize = True
                    
                    # Check for exclusions
                    if op_name in self.excluded_ops:
                        should_quantize = False

                    # Special check for MHA internals
                    if op_name == "MultiheadAttention":
                        # If MHA is requested, we decompose it.
                        # If "Linear" is requested, we also decompose MHA to reach internal Linears.
                        if not should_quantize and "Linear" in self.quantized_ops:
                            # Only re-enable if Linear is NOT excluded
                            if "Linear" not in self.excluded_ops:
                                should_quantize = True
            
            # Check layer config for fp32 override
            if should_quantize and full_name in self.layer_config:
                l_conf = self.layer_config[full_name]
                if l_conf.get('format') == 'fp32' or l_conf.get('type') == 'fp32':
                    should_quantize = False

            if should_quantize:
                QuantClass = OpRegistry.get_quantized_op(type(module))
                new_module = self._create_quantized_module(module, QuantClass, name=full_name)
                
                # If we created a new instance (not in-place swap), we need to set it
                if new_module is not module:
                    setattr(model, name, new_module)
                    
                # Set layer name for tracking/debugging
                # new_module might be the swapped module or the new instance
                target_module = new_module if new_module is not None else module
                target_module.layer_name = full_name
                target_module.run_id = getattr(self, 'run_id', 'default')
            
            # Recursion
            # If we replaced the module, recurse into the new module (e.g. DecomposedMHA)
            # If we swapped in-place, module is the same object but class changed, so recurse into it.
            if new_module is not None:
                self._recursive_replace(new_module, prefix=full_name)
            else:
                self._recursive_replace(module, prefix=full_name)

    def build_model(self, quantized: bool = False) -> nn.Module:
        """Builds and returns the quantized model."""
        # Load base model
        model = self._load_base_model()
        
        if quantized:
            # Fold layers if requested (before quantization)
            if self.fold_layers:
                self._fold_layers(model)

            # Replace layers with quantized versions
            self._replace_layers(model)
            
            # Optimization: Move model to GPU for calibration if available
            # This speeds up the heavy quantize_fp8_generic calls significantly
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model.to(device)
            
            # Calibrate weights (pre-compute scales)
            self._calibrate_model(model)
            
            # FX-based Functional Quantization
            # This handles functional calls like F.relu that are not modules
            try:
                # _fx_quantize returns the modified model (GraphModule) if changes were made
                # FX tracing relies on symbolic execution. GPU vs CPU shouldn't matter for correctness,
                # but we should ensure consistency.
                fx_model = self._fx_quantize(model)
                if fx_model is not None:
                    model = fx_model
            except Exception as e:
                print(f"Warning: FX quantization failed: {e}")
                import traceback
                traceback.print_exc()
        
        return model

    def _fx_quantize(self, model: nn.Module):
        """
        Uses torch.fx to replace functional operations with quantized modules.
        """
        import torch.fx
        import torch.nn.functional as F
        
        # We only want to trace if we have functional ops to replace.
        # Currently we target activations.
        
        # Get supported functional ops from registry
        # We need to map function -> quantized module class
        # OpRegistry stores original_cls -> quantized_cls
        # We need a mapping for functions.
        
        # For now, we hardcode the mapping for common activations as per OpRegistry logic
        # Ideally OpRegistry should support function registration with target class.
        
        func_map = {
            F.relu: "QuantReLU",
            torch.relu: "QuantReLU",
            F.relu6: "QuantReLU6",
            F.gelu: "QuantGELU",
            F.silu: "QuantSiLU",
            F.hardswish: "QuantHardswish",
        }
        
        # Filter by requested quantized ops
        target_funcs = {}
        for func, op_name in func_map.items():
            if "-1" in self.quantized_ops or "all" in self.quantized_ops or op_name in self.quantized_ops:
                # Check exclusions
                if op_name not in self.excluded_ops:
                    target_funcs[func] = op_name

        if not target_funcs:
            return

        # Trace the model
        # We use a custom tracer to ensure we don't trace into existing quantized modules
        class QuantTracer(torch.fx.Tracer):
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                # Treat all registered quantized ops as leaves
                if type(m) in OpRegistry.get_supported_ops().values():
                    return True
                return super().is_leaf_module(m, module_qualified_name)
        
        tracer = QuantTracer()
        try:
            graph = tracer.trace(model)
        except Exception as e:
            print(f"FX Trace failed: {e}")
            return

        gm = torch.fx.GraphModule(model, graph)
        modified = False
        
        for node in list(graph.nodes):
            if node.op == 'call_function' and node.target in target_funcs:
                op_name = target_funcs[node.target]
                QuantClass = OpRegistry.get(op_name)
                
                # Create new module instance
                # We use the same config as the adapter
                kwargs = {'q_type': self.quantization_type}
                if self.quantization_bias is not None:
                    kwargs['quantization_bias'] = self.quantization_bias
                
                # Add specific args if needed (e.g. inplace)
                # We pass them to the module init if supported, or rely on forward kwargs
                # QuantReLU supports inplace in init
                if 'inplace' in node.kwargs:
                    kwargs['inplace'] = node.kwargs['inplace']
                
                new_mod = QuantClass(**kwargs)
                
                # Add module to GraphModule
                new_mod_name = f"{node.name}_quant_{op_name.lower()}"
                gm.add_module(new_mod_name, new_mod)
                
                # Replace node
                with graph.inserting_after(node):
                    new_node = graph.call_module(new_mod_name, args=node.args, kwargs=node.kwargs)
                    node.replace_all_uses_with(new_node)
                
                # Erase old node
                graph.erase_node(node)
                
                modified = True
        
        if modified:
            gm.recompile()
            # We need to update the model reference in the adapter
            # But self.model is set in __init__. build_model returns the model.
            # We are modifying 'model' in-place? No, GraphModule is a new module.
            # We need to return the new GraphModule or update the passed model variable.
            # Since we can't update the local variable 'model' in the caller easily if we just return it,
            # we should copy the state or just return gm.
            
            # However, build_model returns 'model'. We should update 'model' to be 'gm'.
            # But we can't reassign the argument.
            # We will handle this by returning gm from this function and updating in build_model.
            return gm
        
        return model

    def prepare_batch(self, batch):
        """Prepare a batch for model input."""
        images, labels = batch
        
        # We rely on the model's first layer (QuantConv2d) to handle input quantization
        # based on the 'quantize_first_layer' flag.
        # This avoids double quantization and ensures consistent behavior.
            
        return images, labels

    def forward(self, model: nn.Module, batch):
        """Run forward pass."""
        images, _ = batch
        return model(images)

    def get_layer_names(self, model: nn.Module) -> list[str]:
        """Return all module names for layer insertion."""
        return [name for name, _ in model.named_modules()]

    def build_reference_model(self) -> nn.Module:
        """Build a reference (FP32) model for comparison."""
        return self._load_base_model()
