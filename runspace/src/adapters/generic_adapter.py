import torch
import torch.nn as nn
import torchvision.models as models
import os
from .base_adapter import BaseAdapter
from ..registry.op_registry import OpRegistry
from ..quantization.constants import DEFAULT_QUANTIZATION_TYPE
from ..utils.fx_trace_utils import find_non_tensor_nodes, trace_quant_aware
from ..utils.model_input_utils import resolve_model_input_size
try:
    import src.ops # Ensure all ops are registered
except ImportError:
    from .. import ops



class GenericAdapter(BaseAdapter):
    """
    A generic adapter that works with any PyTorch model from torchvision or timm.
    Recursively replaces Conv2d, Linear, BatchNorm2d with quantized ops.
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        model: nn.Module = None,
        model_source: str = "auto",
        weights: str = None,
        input_quantization: bool = True,
        weight_quantization: bool = True,
        output_quantization: bool = False,
        quantize_first_layer: bool = False,
        quantized_ops: list = None,
        excluded_ops: list = None,
        quantization_type: str = DEFAULT_QUANTIZATION_TYPE,
        quantization_bias: int = None,
        layer_config: dict = None,
        per_chunk_format: bool = False,
        input_quantization_type: str = None,
        output_quantization_type: str = None,
        quant_mode: str = "tensor",
        chunk_size: int = None,
        weight_mode: str = "channel",
        weight_chunk_size: int = None,
        act_mode: str = "tensor",
        act_chunk_size: int = None,
        output_mode: str = "tensor",
        output_chunk_size: int = None,
        fold_layers: bool = False,
        simulate_tf32_accum: bool = False,
        rounding: str = "nearest",
        input_chunk_size: int = None,
        run_id: str = "default",
        skip_calibration: bool = False,
        build_quantized: bool = True,
        target_module_prefixes: list = None,
        strict_format_check: bool = False,
        enable_fx_quantization: bool = True,
        unsigned_input_sources: list = None,
        input_size: int | tuple = None,
    ):
        super().__init__()
        self.target_module_prefixes = target_module_prefixes or []
        self.skip_calibration = skip_calibration
        self.build_quantized = bool(build_quantized)
        self.enable_fx_quantization = bool(enable_fx_quantization)
        self.model_name = model_name
        self.base_model_instance = model
        self.model_source = model_source
        self.weights = weights
        self.input_quantization = input_quantization
        self.weight_quantization = weight_quantization
        self.output_quantization = output_quantization
        self.quantize_first_layer = quantize_first_layer
        self.quantized_ops = quantized_ops if quantized_ops is not None else ["all"]
        self.excluded_ops = excluded_ops if excluded_ops is not None else []
        self.quantization_type = quantization_type
        self.quantization_bias = quantization_bias
        self.layer_config = layer_config if layer_config is not None else {}
        self.per_chunk_format = per_chunk_format
        self.input_quantization_type = input_quantization_type
        self.output_quantization_type = output_quantization_type
        self.unsigned_input_sources = unsigned_input_sources or []
        self.run_id = run_id
        self.input_size = input_size

        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.weight_mode = weight_mode
        self.weight_chunk_size = weight_chunk_size
        self.act_mode = act_mode
        self.act_chunk_size = act_chunk_size
        self.output_mode = output_mode
        self.output_chunk_size = output_chunk_size
        self.fold_layers = fold_layers
        self.simulate_tf32_accum = simulate_tf32_accum
        self.rounding = rounding
        self.input_chunk_size = input_chunk_size
        self.strict_format_check = bool(strict_format_check)
        if not self.build_quantized:
            print(
                f"GenericAdapter: build_quantized=False for {self.model_name} "
                "(no quantized ops/layer overrides requested)."
            )
        self.model = self.build_model(quantized=self.build_quantized)

    @property
    def quant_config(self) -> dict:
        """Resolved quantization config consumed by the @observer dispatch
        path so cached Observed* singletons get configured with the same
        format/mode/chunk_size as in-tree quantized modules."""
        return {
            "format": self.quantization_type,
            "mode": self.quant_mode,
            "chunk_size": self.input_chunk_size if self.input_chunk_size is not None else self.chunk_size,
            "quantization_bias": self.quantization_bias,
        }

    def _auto_detect_model_source(self) -> str:
        """Detect whether the model lives in torchvision or timm."""
        if hasattr(models, self.model_name):
            return "torchvision"
        try:
            import timm
            if timm.is_model(self.model_name):
                return "timm"
        except ImportError:
            pass
        raise ValueError(
            f"Model '{self.model_name}' not found in torchvision or timm. "
            f"Specify model_source explicitly if using a custom source."
        )

    def _load_base_model(self) -> nn.Module:
        """Loads the base model from torchvision or other sources."""
        if self.base_model_instance is not None:
            import copy
            return copy.deepcopy(self.base_model_instance)

        if self.model_source == "auto":
            self.model_source = self._auto_detect_model_source()
            print(f"Auto-detected model source: {self.model_source}")

        if self.model_source == "torchvision":
            return self._load_torchvision_model()
        elif self.model_source == "timm":
            return self._load_timm_model()
        else:
            raise ValueError(f"Unknown model_source: '{self.model_source}'. Use 'torchvision', 'timm', or 'auto'.")

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
            # Check if weights is a path to a file
            if isinstance(self.weights, str) and os.path.isfile(self.weights):
                print(f"Loading custom weights from {self.weights}...")
                # Load model without weights first
                model = model_fn(weights=None)
                try:
                    self._load_custom_weights_into_model(model, self.weights)
                    return model
                except Exception as e:
                    raise RuntimeError(f"Failed to load weights from {self.weights}: {e}")

            # Resolve torchvision weights enum robustly (including ViT).
            weights_enum = self._resolve_torchvision_weights(model_fn)
            model = model_fn(weights=None)
            if weights_enum is not None:
                try:
                    # Avoid torchvision constructor-side weight-loading path.
                    # Load the resolved checkpoint into a skeleton model directly.
                    state_dict = weights_enum.get_state_dict(progress=False)
                    incompatible = model.load_state_dict(state_dict, strict=True)
                    if incompatible is not None:
                        missing = list(getattr(incompatible, 'missing_keys', []) or [])
                        unexpected = list(getattr(incompatible, 'unexpected_keys', []) or [])
                        if missing or unexpected:
                            print(
                                f"Torchvision load notes for {self.model_name}: "
                                f"missing={len(missing)}, unexpected={len(unexpected)}"
                            )
                    return model
                except Exception as e:
                    print(
                        f"Warning: direct state_dict load for {self.model_name} "
                        f"failed ({e}); continuing with weights=None."
                    )
                    return model
            print(
                f"Warning: Could not resolve torchvision weights '{self.weights}' for "
                f"{self.model_name}; falling back to weights=None."
            )
            return model
        else:
            return model_fn(weights=None)

    def _resolve_torchvision_weights(self, model_fn):
        """
        Resolve a torchvision weights spec into a proper enum instance.
        Avoids deprecated `weights=True` fallback paths.
        """
        weights_spec = self.weights
        if weights_spec is None:
            return None

        # Already an enum/object: trust caller.
        if not isinstance(weights_spec, (str, bool, int)):
            return weights_spec

        token = str(weights_spec).strip()
        token_l = token.lower()
        if token_l in ("", "none", "false", "0", "null"):
            return None

        wants_default = token_l in ("default", "true", "1")

        # Preferred path: torchvision's official model-specific resolver.
        try:
            from torchvision.models import get_model_weights
            weights_cls = get_model_weights(model_fn)
            if wants_default and hasattr(weights_cls, "DEFAULT"):
                return weights_cls.DEFAULT

            if hasattr(weights_cls, token):
                return getattr(weights_cls, token)

            token_up = token.upper()
            for attr in dir(weights_cls):
                if attr.upper() == token_up:
                    return getattr(weights_cls, attr)

            if not wants_default and hasattr(weights_cls, "DEFAULT"):
                print(
                    f"Warning: Unknown torchvision weights token '{token}' for "
                    f"{self.model_name}; using DEFAULT."
                )
                return weights_cls.DEFAULT
        except Exception:
            pass

        # Legacy mapping fallback.
        mapped = self._get_weights_enum()
        if mapped is not None:
            return mapped

        return None

    @staticmethod
    def _extract_checkpoint_state_dict(checkpoint):
        """Extract tensor state_dict from common checkpoint container layouts."""
        if not isinstance(checkpoint, dict):
            return checkpoint

        for key in ('state_dict', 'model', 'model_state_dict', 'state_dict_ema'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        return checkpoint

    @staticmethod
    def _strip_common_prefixes(state_dict):
        """Strip common wrappers (DataParallel / trainer containers) from keys."""
        if not isinstance(state_dict, dict) or not state_dict:
            return state_dict

        normalized = state_dict
        for prefix in ('module.', 'model.'):
            keys = list(normalized.keys())
            if keys and all(isinstance(k, str) and k.startswith(prefix) for k in keys):
                normalized = {k[len(prefix):]: v for k, v in normalized.items()}
        return normalized

    def _load_custom_weights_into_model(self, model: nn.Module, weights_path: str):
        """Load external checkpoint into model with tolerant key handling."""
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = self._extract_checkpoint_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Checkpoint at {weights_path} did not contain a valid state_dict mapping."
            )

        state_dict = self._strip_common_prefixes(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)

        # Helpful diagnostics without failing hard here; runner validation enforces correctness.
        if incompatible is not None:
            missing = list(getattr(incompatible, 'missing_keys', []) or [])
            unexpected = list(getattr(incompatible, 'unexpected_keys', []) or [])
            if missing or unexpected:
                print(
                    f"Checkpoint load notes for {weights_path}: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

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
            "vit_b_16": "ViT_B_16_Weights",
            "vit_b_32": "ViT_B_32_Weights",
            "vit_l_16": "ViT_L_16_Weights",
            "vit_l_32": "ViT_L_32_Weights",
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

    def _load_timm_model(self) -> nn.Module:
        """Load a model from the timm library."""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "The 'timm' package is required to load this model. "
                "Install it with: pip install timm"
            )

        custom_weight_file = isinstance(self.weights, str) and os.path.isfile(self.weights)
        pretrained = (
            bool(self.weights)
            and str(self.weights).lower() not in ('none', 'false', '0')
            and not custom_weight_file
        )
        if not timm.is_model(self.model_name):
            raise ValueError(
                f"Model '{self.model_name}' not found in timm. "
                f"Check available models with: timm.list_models('{self.model_name}*')"
            )

        create_kwargs = {}
        if self.input_size is not None:
            create_kwargs['img_size'] = self.input_size

        if custom_weight_file:
            print(f"Loading custom timm weights from {self.weights}...")
            model = timm.create_model(self.model_name, pretrained=False, **create_kwargs)
            try:
                self._load_custom_weights_into_model(model, self.weights)
            except Exception as e:
                raise RuntimeError(f"Failed to load weights from {self.weights}: {e}")
            return model

        model = timm.create_model(self.model_name, pretrained=pretrained, **create_kwargs)
        return model

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

        model.eval()

        modules_to_fuse = []
        named_modules = list(model.named_modules())

        i = 0
        while i < len(named_modules) - 1:
            name_curr, mod_curr = named_modules[i]
            name_next, mod_next = named_modules[i+1]

            if (
                isinstance(mod_curr, nn.Conv2d) and isinstance(mod_next, nn.BatchNorm2d)
                or isinstance(mod_curr, nn.Linear) and isinstance(mod_next, nn.BatchNorm1d)
            ):
                modules_to_fuse.append([name_curr, name_next])
                i += 2
                continue

            i += 1

        if modules_to_fuse:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

    def _replace_layers(self, model: nn.Module):
        """Replace standard layers with quantized versions, then FX-rewrite
        inline functional ops (torch.cat / +/torch.einsum / F.softmax / ...)
        inside any FX-traceable composite, then propagate config to all
        QuantizedLayerMixin instances (in-tree class swaps and the FX-
        attached submodules alike)."""
        self._first_layer_found = False
        self._recursive_replace(model)
        if self.enable_fx_quantization:
            try:
                self._fx_quantize(model)
            except Exception as e:
                print(f"Note: FX quantization skipped ({type(e).__name__}: {e})")
        self._configure_remaining_mixin_ops(model)

    def _configure_remaining_mixin_ops(self, model: nn.Module):
        """Propagate adapter config + per-layer overrides to QuantizedLayerMixin ops
        that weren't touched by _recursive_replace. Covers Observed* ops registered
        without original_cls (ObservedAttentionScores / Apply / Concat / Add /
        DescMatmul) whose q_type would otherwise stay at the ctor default regardless
        of what config.yaml says."""
        # Match the absolute prefix SuperGlue uses ([superglue.py:51]) so the
        # QuantizedLayerMixin class identity is the same as the one Observed* ops
        # inherit from. A relative import would bind to src.ops.quant_base (a
        # separate module when both runspace/ and the repo root are on sys.path).
        from runspace.src.ops.quant_base import QuantizedLayerMixin

        supported_ops_values = tuple(OpRegistry.get_supported_ops().values())

        for full_name, module in model.named_modules():
            if not isinstance(module, QuantizedLayerMixin):
                continue
            # Already configured by Case 1/2 (class is a registered quantized op).
            if type(module) in supported_ops_values:
                continue

            settings = self._layer_quant_settings(full_name)
            module.q_type = settings['q_type']
            module.quantization_bias = settings['bias']
            module.input_q_type = self._effective_input_q_type(settings)
            module.input_quantization = self.input_quantization
            module.weight_quantization = self.weight_quantization
            module.input_mode = settings['input_mode']
            module.input_chunk_size = settings['input_chunk_size']
            module.output_q_type = self._effective_output_q_type(settings)
            module.output_quantization = settings['output_quantization']
            module.output_mode = settings['output_mode']
            module.output_chunk_size = settings['output_chunk_size']
            module.rounding = settings['rounding']
            module.layer_name = full_name
            module.run_id = getattr(self, 'run_id', 'default')

    def _layer_quant_settings(self, name: str) -> dict:
        """Resolve global quantization settings plus per-layer overrides."""
        layer_conf = self.layer_config.get(name, {}) if isinstance(self.layer_config, dict) else {}
        q_type = layer_conf.get('type', layer_conf.get('format', self.quantization_type))

        # Output-quantization on/off precedence:
        # 1. explicit per-layer `output_quantization: true|false` wins
        # 2. presence of any per-layer output_* key implicitly enables it
        # 3. otherwise inherit from the global adapter flag
        if 'output_quantization' in layer_conf:
            layer_output_quant = bool(layer_conf['output_quantization'])
        elif any(k in layer_conf for k in ('output_format', 'output_mode', 'output_chunk_size')):
            layer_output_quant = True
        else:
            layer_output_quant = self.output_quantization

        return {
            'layer_conf': layer_conf,
            'q_type': q_type,
            'bias': layer_conf.get('bias', self.quantization_bias),
            'input_q_type': layer_conf.get('input_format', self.input_quantization_type),
            'input_mode': layer_conf.get('mode', self.quant_mode),
            'input_chunk_size': layer_conf.get(
                'chunk_size',
                self.input_chunk_size if self.input_chunk_size is not None else self.chunk_size,
            ),
            'weight_mode': layer_conf.get('weight_mode', self.weight_mode),
            'weight_chunk_size': layer_conf.get('weight_chunk_size', self.weight_chunk_size),
            'act_mode': layer_conf.get('act_mode', self.act_mode),
            'act_chunk_size': layer_conf.get('act_chunk_size', self.act_chunk_size),
            'output_quantization': layer_output_quant,
            'output_q_type': layer_conf.get('output_format', self.output_quantization_type),
            'output_mode': layer_conf.get('output_mode', self.output_mode),
            'output_chunk_size': layer_conf.get('output_chunk_size', self.output_chunk_size),
            'rounding': layer_conf.get('rounding', self.rounding),
        }

    @staticmethod
    def _effective_input_q_type(settings: dict) -> str:
        return settings['input_q_type'] if settings['input_q_type'] else settings['q_type']

    @staticmethod
    def _effective_output_q_type(settings: dict) -> str:
        return settings['output_q_type'] if settings['output_q_type'] else settings['q_type']

    @staticmethod
    def _contains_requested_name(configured_names, candidate_names) -> bool:
        lowered = {name.lower() for name in configured_names}
        return any(candidate.lower() in lowered for candidate in candidate_names)

    @staticmethod
    def _is_timm_attention_like(module: nn.Module) -> bool:
        return (
            hasattr(module, "qkv") and isinstance(getattr(module, "qkv"), nn.Linear) and
            hasattr(module, "proj") and isinstance(getattr(module, "proj"), nn.Linear) and
            hasattr(module, "num_heads")
        )

    @staticmethod
    def _is_timm_mlp_like(module: nn.Module) -> bool:
        return (
            hasattr(module, "fc1") and isinstance(getattr(module, "fc1"), nn.Linear) and
            hasattr(module, "fc2") and isinstance(getattr(module, "fc2"), nn.Linear)
        )

    def _create_quantized_module(self, module: nn.Module, QuantClass: type, name: str = "") -> nn.Module:
        """Creates a quantized module from the original module."""
        from ..ops.quant_base import QuantizedLayerMixin
        try:
            from runspace.src.ops.quant_base import QuantizedLayerMixin as RunspaceQuantizedLayerMixin
        except ImportError:
            RunspaceQuantizedLayerMixin = QuantizedLayerMixin
        quantized_mixin_types = tuple(dict.fromkeys((QuantizedLayerMixin, RunspaceQuantizedLayerMixin)))
        settings = self._layer_quant_settings(name)
        q_type = settings['q_type']
        bias = settings['bias']
        layer_conf = settings['layer_conf']
        
        # Case 1: Custom from_native factory (e.g. MHA, BasicConv2d)
        if hasattr(QuantClass, 'from_native'):
             created = QuantClass.from_native(module, q_type=q_type, quantization_bias=bias)
             if isinstance(created, quantized_mixin_types):
                 created.q_type = q_type
                 created.quantization_bias = bias
                 created.input_q_type = self._effective_input_q_type(settings)
                 created.input_quantization = self.input_quantization
                 created.weight_quantization = self.weight_quantization
                 created.input_mode = settings['input_mode']
                 created.input_chunk_size = settings['input_chunk_size']
                 created.weight_mode = settings['weight_mode']
                 created.weight_chunk_size = settings['weight_chunk_size']
                 created.output_q_type = self._effective_output_q_type(settings)
                 created.output_quantization = settings['output_quantization']
                 created.output_mode = settings['output_mode']
                 created.output_chunk_size = settings['output_chunk_size']
                 created.rounding = settings['rounding']
             return created

        # Case 2: Layer with weights (Conv, Linear, BN) - In-place class swap.
        # Activations are routed to Case 3 because their __init__ builds
        # per-class state (e.g. QuantGELU.piecewise_lut, QuantSoftmax.uq_type)
        # that an in-place __class__ swap would skip.
        if issubclass(QuantClass, quantized_mixin_types) and not OpRegistry.is_activation(QuantClass.__name__):
            # Perform in-place class swap to preserve weights
            module.__class__ = QuantClass

            # Initialize Mixin state
            module.q_type = q_type
            module.quantization_bias = bias
            module.input_q_type = self._effective_input_q_type(settings)

            module.input_quantization = self.input_quantization
            module.weight_quantization = self.weight_quantization
            module.input_mode = settings['input_mode']
            module.input_chunk_size = settings['input_chunk_size']
            module.weight_mode = settings['weight_mode']
            module.weight_chunk_size = settings['weight_chunk_size']
            module.output_q_type = self._effective_output_q_type(settings)
            module.output_quantization = settings['output_quantization']
            module.output_mode = settings['output_mode']
            module.output_chunk_size = settings['output_chunk_size']
            module.rounding = settings['rounding']

            # Per-chunk format configuration
            if self.per_chunk_format and 'chunk_formats' in layer_conf:
                module.chunk_formats = layer_conf['chunk_formats']

            module.register_buffer('weight_scale', None)
            module.register_buffer('weight_fp8', None)

            # Set TF32 simulation flag if supported
            if hasattr(module, 'simulate_tf32_accum') or QuantClass.__name__ in ("QuantConv2d", "QuantConv1d"):
                module.simulate_tf32_accum = self.simulate_tf32_accum

            # Handle first layer logic for conv layers
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                is_first = not self._first_layer_found
                if is_first:
                    self._first_layer_found = True
                module.is_first_layer = is_first
                if isinstance(module, nn.Conv2d) and module.in_channels == 3 and is_first:
                    module.quantize_first_layer = self.quantize_first_layer
            
            return module

        # Case 3: Activations / Softmax - New instance
        kwargs = {'q_type': q_type}
        if bias is not None:
            kwargs['quantization_bias'] = bias
            
        # Pass input mode params to activation (activations only have inputs)
        kwargs['quant_mode'] = settings['act_mode']
        kwargs['chunk_size'] = settings['act_chunk_size']
        kwargs['unsigned_input_sources'] = self.unsigned_input_sources
        
        # Copy common attributes if they exist
        if hasattr(module, 'inplace'):
            kwargs['inplace'] = module.inplace
        if hasattr(module, 'approximate'):
            kwargs['approximate'] = module.approximate
        if hasattr(module, 'dim'):
            kwargs['dim'] = module.dim
            
        created = QuantClass(**kwargs)
        # Some activation __init__s accept q_type as a kwarg but don't assign
        # it to self (e.g. QuantGELU, QuantReLU). Without this assignment,
        # the report's hasattr(module, 'q_type') check would fall through and
        # mark Output Q? / Output Fmt as N/A even though they're well-defined.
        created.q_type = q_type
        if bias is not None:
            created.quantization_bias = bias
        created.input_quantization = self.input_quantization
        created.input_q_type = self._effective_input_q_type(settings)
        created.output_quantization = settings['output_quantization']
        created.output_q_type = self._effective_output_q_type(settings)
        created.output_mode = settings['output_mode']
        created.output_chunk_size = settings['output_chunk_size']
        created.rounding = settings['rounding']
        return created

    def _recursive_replace(self, model: nn.Module, prefix: str = ""):
        """Recursively traverse and replace layers using the registry."""
        quantized_ops_lc = {o.lower() for o in self.quantized_ops}
        excluded_ops_lc = {o.lower() for o in self.excluded_ops}
        quantize_all = "-1" in quantized_ops_lc or "all" in quantized_ops_lc

        # Get supported ops from registry
        supported_ops = OpRegistry.get_supported_ops()

        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            new_module = None
            should_quantize = False
            quant_class = None

            # If target_module_prefixes is set, only quantize layers whose full path
            # starts with one of the declared prefixes; others are recursed but skipped.
            if self.target_module_prefixes and not any(
                full_name.startswith(p) or p.startswith(full_name)
                for p in self.target_module_prefixes
            ):
                self._recursive_replace(module, prefix=full_name)
                continue
            matched_op_name = module.__class__.__name__
            matched_original_name = None

            # Do not try to "re-quantize" modules that are already quantized.
            if type(module) in supported_ops.values():
                self._recursive_replace(module, prefix=full_name)
                continue
            
            # Check if module type is supported (subclass-aware)
            for original_cls, q_cls in supported_ops.items():
                if isinstance(module, original_cls):
                    quant_class = q_cls
                    matched_original_name = original_cls.__name__
                    break

            if quant_class is not None:
                requested_names = {matched_op_name}
                if matched_original_name is not None:
                    requested_names.add(matched_original_name)

                if quantize_all or self._contains_requested_name(quantized_ops_lc, requested_names):
                    should_quantize = True

                # If Linear is requested, decompose MHA as well to expose q/k/v internals.
                if isinstance(module, nn.MultiheadAttention) and "linear" in quantized_ops_lc and "linear" not in excluded_ops_lc:
                    should_quantize = True

                if self._contains_requested_name(excluded_ops_lc, requested_names):
                    should_quantize = False

                # timm uses BatchNormAct2d (subclass of nn.BatchNorm2d) in many
                # CNN stems/blocks. In-place swapping that subclass to
                # QuantBatchNorm2d drops its fused drop/activation behavior and
                # breaks model math. Keep the native wrapper intact.
                if (
                    should_quantize and
                    isinstance(module, nn.BatchNorm2d) and
                    type(module) is not nn.BatchNorm2d and
                    (hasattr(module, "act") or hasattr(module, "drop"))
                ):
                    from ..ops.quant_bn import QuantBatchNormAct2d
                    quant_class = QuantBatchNormAct2d
                    should_quantize = True

            # timm-specific Attention / MLP decomposition
            if not should_quantize and self._is_timm_attention_like(module):
                wants_attention = quantize_all or self._contains_requested_name(
                    quantized_ops_lc,
                    {"Attention", "MultiheadAttention", "MHA", "QkvAttention"}
                )
                wants_internal_linear = "linear" in quantized_ops_lc and "linear" not in excluded_ops_lc
                if wants_attention or wants_internal_linear:
                    from ..ops.quant_mha import DecomposedQkvAttention
                    quant_class = DecomposedQkvAttention
                    should_quantize = True

            if not should_quantize and self._is_timm_mlp_like(module):
                wants_mlp = quantize_all or self._contains_requested_name(
                    quantized_ops_lc,
                    {"Mlp", "MLP", "FeedForward", "FFN", "MlpBlock"}
                )
                wants_internal_linear = "linear" in quantized_ops_lc and "linear" not in excluded_ops_lc
                if wants_mlp or wants_internal_linear:
                    from ..ops.quant_mha import DecomposedMlpBlock
                    quant_class = DecomposedMlpBlock
                    should_quantize = True
            
            # Check layer config for fp32 override
            if should_quantize and full_name in self.layer_config:
                l_conf = self.layer_config[full_name]
                if l_conf.get('format') == 'fp32' or l_conf.get('type') == 'fp32':
                    should_quantize = False

            if should_quantize:
                new_module = self._create_quantized_module(module, quant_class, name=full_name)
                
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
            
            # Optimization: Move model to GPU only when calibrating.
            # For skip_calibration flows (file-backed load), keep model on CPU here
            # and let runner move it after explicit state_dict loading.
            if torch.cuda.is_available() and not self.skip_calibration:
                device = torch.device('cuda')
                model.to(device)
            
            # Calibrate weights (pre-compute scales) — skip when weights will be
            # loaded from a pre-quantized cache immediately after construction.
            if not self.skip_calibration:
                self._calibrate_model(model)
            
            # FX-based Functional Quantization
            # This handles functional calls like F.relu that are not modules.
            # Gated to custom-source models only — for stock torchvision/timm
            # backbones it would attach QuantAdd/QuantCat/etc. submodules
            # for inline `+` and `torch.cat` calls, which appear in the
            # compliance report as non-original-module rows (e.g. `add_7`).
            if self.enable_fx_quantization:
                try:
                    # _fx_quantize returns the modified model (GraphModule) if changes were made
                    # FX tracing relies on symbolic execution. GPU vs CPU shouldn't matter for correctness,
                    # but we should ensure consistency.
                    fx_model = self._fx_quantize(model)
                    if fx_model is not None:
                        model = fx_model
                except Exception as e:
                    # Control-flow models (e.g. timm MobileViT) can't be FX-traced — that's fine,
                    # recursive replacement already handled all registered ops.
                    print(f"Note: FX quantization skipped ({type(e).__name__}: {e})")
                    raise e # Do not remove
                    
            if self.unsigned_input_sources:
                self._propagate_unsigned_inputs(model)
        
        return model

    def _propagate_unsigned_inputs(self, model: nn.Module):
        """
        Propagates the 'unsigned' input property to layers that follow unsigned sources.
        """
        unsigned_sources_lc = {s.lower() for s in self.unsigned_input_sources}
        if not unsigned_sources_lc:
            return model
        
        from ..ops.quant_softmax import qtype_to_unsigned_qtype

        def _process_graph_module(gm: torch.fx.GraphModule):
            modified_any = False
            # First pass: identify unsigned nodes
            unsigned_nodes = set()
            passthrough_ops_lc = {'dropout', 'quantdropout'}
            
            # Map node name to the module instance for easier access
            modules = dict(gm.named_modules())
            
            for node in gm.graph.nodes:
                is_unsigned_source = False
                is_passthrough = False
                
                if node.op == 'call_module':
                    target_mod = modules.get(node.target)
                    if target_mod is not None:
                        class_name = target_mod.__class__.__name__.lower()
                        if class_name in unsigned_sources_lc or class_name.replace('quant', '') in unsigned_sources_lc:
                            is_unsigned_source = True
                        elif class_name in passthrough_ops_lc or class_name.replace('quant', '') in passthrough_ops_lc:
                            is_passthrough = True
                elif node.op == 'call_function':
                    # target for call_function can be a string (from builtins) or the function itself
                    if isinstance(node.target, str):
                        func_name = node.target.lower()
                    else:
                        func_name = node.target.__name__.lower()
                        
                    if func_name in unsigned_sources_lc:
                        is_unsigned_source = True
                    elif func_name in passthrough_ops_lc:
                        is_passthrough = True
                
                if is_unsigned_source:
                    unsigned_nodes.add(node.name)
                elif is_passthrough:
                    # Passthrough nodes inherit 'unsigned' if their input is unsigned
                    node_inputs = [arg for arg in node.args if isinstance(arg, torch.fx.Node)]
                    if node_inputs and any(inp.name in unsigned_nodes for inp in node_inputs):
                        unsigned_nodes.add(node.name)
            
            # Second pass: update inputs
            for node in gm.graph.nodes:
                if node.op == 'call_module':
                    target_mod = modules.get(node.target)
                    if target_mod is not None:
                        # Extract all positional nodes (most common for activations/weights)
                        node_inputs = [arg for arg in node.args if isinstance(arg, torch.fx.Node)]
                        
                        # Handle multi-input modules (MatMul, Add, etc.)
                        if len(node_inputs) > 1:
                            for idx, inp in enumerate(node_inputs):
                                if inp.name in unsigned_nodes:
                                    attr_name = f'input{idx+1}_q_type'
                                    # Fallback to general input_q_type or q_type
                                    base_qtype = getattr(target_mod, attr_name, 
                                                       getattr(target_mod, 'input_q_type', 
                                                              getattr(target_mod, 'q_type', 'fp8_e4m3')))
                                    setattr(target_mod, attr_name, qtype_to_unsigned_qtype(base_qtype))
                                    modified_any = True
                                    
                        # Handle single-input modules
                        elif len(node_inputs) == 1 and node_inputs[0].name in unsigned_nodes:
                            if hasattr(target_mod, 'input_q_type'):
                                target_mod.input_q_type = qtype_to_unsigned_qtype(target_mod.input_q_type)
                                modified_any = True
                            elif hasattr(target_mod, 'q_type'):
                                # Fallback if only q_type exists but no explicit input_q_type
                                # We set input_q_type to avoid changing weight q_type
                                setattr(target_mod, 'input_q_type', qtype_to_unsigned_qtype(target_mod.q_type))
                                modified_any = True
                                
            return modified_any

        # If model is already a GraphModule (e.g. from _fx_quantize), use it directly
        if isinstance(model, torch.fx.GraphModule):
            if _process_graph_module(model):
                print(f"Propagated unsigned input formats using existing GraphModule.")
            return model

        # Try whole model trace
        try:
            _, _, gm = trace_quant_aware(model)
            if _process_graph_module(gm):
                print(f"Propagated unsigned input formats using full-model FX trace.")
            return model
        except Exception as e:
            pass
            
        # Fall back to submodules
        modified = False
        for child_name, child in model.named_children():
            if self._is_safe_fx_submodule(child):
                try:
                    _, _, gm = trace_quant_aware(child)
                    if _process_graph_module(gm):
                        modified = True
                except Exception:
                    pass
        if modified:
             print(f"Propagated unsigned input formats using submodule FX trace.")
             
        return model


    def _fx_quantize(self, model: nn.Module):
        """
        Uses torch.fx to replace functional operations with quantized modules.
        """
        import torch.nn.functional as F
        
        # We only want to trace if we have functional ops to replace.
        # Currently we target activations.
        
        # Get supported functional ops from registry
        # We need to map function -> quantized module class
        # OpRegistry stores original_cls -> quantized_cls
        # We need a mapping for functions.
        
        # For now, we hardcode the mapping for common activations as per OpRegistry logic
        # Ideally OpRegistry should support function registration with target class.
        
        import operator
        func_map = {
            F.relu: "QuantReLU",
            torch.relu: "QuantReLU",
            F.relu6: "QuantReLU6",
            F.gelu: "QuantGELU",
            F.silu: "QuantSiLU",
            F.hardswish: "QuantHardswish",
            torch.matmul: "QuantMatMul",
            operator.matmul: "QuantMatMul",
            torch.bmm: "QuantBMM",
            torch.add: "QuantAdd",
            operator.add: "QuantAdd",
            torch.sub: "QuantSub",
            operator.sub: "QuantSub",
            torch.mul: "QuantMul",
            operator.mul: "QuantMul",
            torch.div: "QuantDiv",
            operator.truediv: "QuantDiv",
            torch.cat: "QuantCat",
        }
        
        # Filter by requested quantized ops
        quantized_ops_lc = {o.lower() for o in self.quantized_ops}
        excluded_ops_lc = {o.lower() for o in self.excluded_ops}
        target_funcs = {}
        try:
            for func, op_name in func_map.items():
                op_name_lc = op_name.lower()
                bare_lc = op_name_lc[len("quant"):] if op_name_lc.startswith("quant") else op_name_lc
                if "-1" in quantized_ops_lc or "all" in quantized_ops_lc or op_name_lc in quantized_ops_lc or bare_lc in quantized_ops_lc:
                    if op_name_lc not in excluded_ops_lc and bare_lc not in excluded_ops_lc:
                        target_funcs[func] = op_name
        except Exception as e:
            print(f"1.Error in FX quantization: {e}")
        if not target_funcs:
            return

        try:
            try:
                fx_model, modified = self._fx_quantize_module_graph(model, target_funcs)
            except Exception as e:
                print(f"3.Error in FX quantization: {e}")
                return None
            if modified and fx_model is not None:
                valid, error = self._validate_fx_model(fx_model, model)
                if valid:
                    return fx_model
                print(
                    "Note: FX whole-model quantization produced an invalid runtime graph "
                    f"({type(error).__name__}: {error}). Falling back to submodule FX."
                )
        except Exception as e:
            print(f"2.Error in FX quantization: {e}") 
            return None

        submodule_modified = self._fx_quantize_submodules(model, target_funcs)
        if submodule_modified:
            return model

        if modified and fx_model is not None:
            print("Note: submodule FX fallback did not find any safe traceable children.")
        return model

    def _fx_quantize_module_graph(self, module: nn.Module, target_funcs: dict):
        try:
            _, graph, gm = trace_quant_aware(module)
        except Exception as e:
            return None, False
        non_tensor_nodes = find_non_tensor_nodes(graph)
        modified = False
        
        for node in list(graph.nodes):
            if node.name in non_tensor_nodes:
                continue
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

                # Add module to GraphModule. Use FX's auto-numbered node.name
                # (e.g. "cat", "add", "add_1", "truediv") so the report path
                # reads cleanly: backbone.superglue.gnn.cat instead of
                # backbone.superglue.gnn.cat_quant_quantcat.
                base = node.name
                new_mod_name = base
                _i = 0
                while hasattr(gm, new_mod_name):
                    _i += 1
                    new_mod_name = f"{base}_{_i}"
                gm.add_module(new_mod_name, new_mod)

                # Propagate runtime config (mirrors _create_quantized_module for
                # in-place class swaps).  Without these, mode='chunk' layers hit
                # input_chunk_size=None at the codec entry guard.
                fx_settings = self._layer_quant_settings(new_mod_name)
                new_mod.input_q_type = self._effective_input_q_type(fx_settings)
                new_mod.input_quantization = self.input_quantization
                new_mod.weight_quantization = self.weight_quantization
                new_mod.input_mode = fx_settings['input_mode']
                new_mod.input_chunk_size = fx_settings['input_chunk_size']
                new_mod.weight_mode = fx_settings['weight_mode']
                new_mod.weight_chunk_size = fx_settings['weight_chunk_size']
                new_mod.output_q_type = self._effective_output_q_type(fx_settings)
                new_mod.output_quantization = fx_settings['output_quantization']
                new_mod.output_mode = fx_settings['output_mode']
                new_mod.output_chunk_size = fx_settings['output_chunk_size']
                new_mod.rounding = fx_settings['rounding']
                
                # Replace node
                with graph.inserting_after(node):
                    new_node = graph.call_module(new_mod_name, args=node.args, kwargs=node.kwargs)
                    node.replace_all_uses_with(new_node)
                
                # Erase old node
                graph.erase_node(node)
                
                modified = True
        
        if modified:
            for attr in ("default_cfg", "pretrained_cfg"):
                if hasattr(module, attr):
                    setattr(gm, attr, getattr(module, attr))
            gm.recompile()
        return gm, modified

    def _validate_fx_model(self, fx_model: nn.Module, reference_model: nn.Module):
        # The CUDA codec is the only supported backend for the standard
        # `quantize_tensor` path, so the validation forward must run on CUDA
        # even when the model is still on CPU at build time (the runner
        # moves it to self.device after build_model() returns).
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        try:
            fx_model.to(device)
            dummy = self._get_dummy_input(fx_model).to(device)
            with torch.no_grad():
                fx_model.eval()(dummy)
            return True, None
        except Exception as e:
            return False, e

    def _get_dummy_input(self, model: nn.Module) -> torch.Tensor:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        _, c, h, w = resolve_model_input_size(model)
        return torch.zeros(1, c, h, w, device=device)

    def _is_safe_fx_submodule(self, module: nn.Module) -> bool:
        if isinstance(module, nn.Sequential):
            return True

        cls = type(module)
        module_name = cls.__module__

        # Skip standard torch.nn.* leaves (Conv, Linear, BN, ReLU, MaxPool, ...)
        # and already-quantized swaps — neither has inline ops to rewrite.
        if module_name.startswith("torch.nn"):
            return False
        from runspace.src.ops.quant_base import QuantizedLayerMixin
        if isinstance(module, QuantizedLayerMixin):
            return False
        if module_name == "runspace.src.ops.observed_ops":
            return False
        # Need children — leaf custom modules can't host inline ops to rewrite.
        if next(iter(module.children()), None) is None:
            return False

        # Accept any remaining custom composite (timm Block/Attention/Mlp,
        # SuperGlue's KeypointEncoder/MultiHeadedAttention/AttentionalPropagation,
        # user models). Non-traceable forwards get filtered downstream when
        # symbolic_trace raises.
        return True

    def _fx_quantize_submodules(self, module: nn.Module, target_funcs: dict) -> bool:
        modified_any = False

        for child_name, child in list(module.named_children()):
            if self._is_safe_fx_submodule(child):
                quantized_child, modified = self._fx_quantize_module_graph(child, target_funcs)
                if modified and quantized_child is not None:
                    setattr(module, child_name, quantized_child)
                    modified_any = True
                    continue

            if self._fx_quantize_submodules(child, target_funcs):
                modified_any = True

        return modified_any

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

    def create_metrics(self):
        """Returns a MetricsEngine for classification/SLM evaluation."""
        from src.eval.metrics import MetricsEngine
        return MetricsEngine()

    def build_reference_model(self) -> nn.Module:
        """Build a reference (FP32) model for comparison.

        If the runner's materialize-weights flow clobbered self.weights to None
        (so the adapter builds a skeleton and loads from a .pt), the runner
        stashes the original spec on `self._reference_weights_spec`. Use it so
        the reference model gets real pretrained weights, not random init.
        """
        ref_weights = getattr(self, '_reference_weights_spec', None) or self.weights
        saved = self.weights
        try:
            self.weights = ref_weights
            return self._load_base_model()
        finally:
            self.weights = saved
