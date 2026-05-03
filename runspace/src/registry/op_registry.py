import torch.nn as nn
import sys

if __name__ == "src.registry.op_registry":
    sys.modules.setdefault("runspace.src.registry.op_registry", sys.modules[__name__])
elif __name__ == "runspace.src.registry.op_registry":
    sys.modules.setdefault("src.registry.op_registry", sys.modules[__name__])

class OpRegistry:
    """
    A central mapping from logical ops to implementations.
    """
    _registry = {}
    _supported_ops = {} # Mapping from original_cls -> quantized_cls
    _activation_ops = set() # Set of quantized op names that are activations
    _compliance_status = {} # Mapping from op_name -> status message (if custom)
    _supported_functions = set() # Set of supported functional operations (e.g. F.conv2d)
    _under_construction_ops = set() # Set of ops marked as under construction
    _passthrough_ops = set() # Ops that forward to the next quantized layer without W/A quant of their own
    _replacements_by_name = {} # Maps upstream @observer function __name__ -> (Observed cls, init_from_args dict)
    _unquantized_ops = set() # Ops registered with quantized=False — observed but no actual W/A quantization

    @classmethod
    def register(cls, op_name: str, original_cls=None, *, replaces=None, init_from_args=None,
                 is_activation=False, compliance_status=None, under_construction=False, passthrough=False,
                 quantized=True):
        def decorator(cls_impl):
            cls._registry[op_name] = cls_impl
            if original_cls:
                cls._supported_ops[original_cls] = cls_impl
            if replaces is not None:
                cls._replacements_by_name[replaces] = (cls_impl, dict(init_from_args or {}))
            if is_activation:
                cls._activation_ops.add(op_name)
            if compliance_status:
                cls._compliance_status[op_name] = compliance_status
            if under_construction:
                cls._under_construction_ops.add(op_name)
            if passthrough:
                cls._passthrough_ops.add(op_name)
            if not quantized:
                cls._unquantized_ops.add(op_name)
            return cls_impl
        return decorator

    @classmethod
    def register_function(cls, func):
        """Registers a functional operation as supported."""
        cls._supported_functions.add(func)

    @classmethod
    def get(cls, op_name: str):
        if op_name not in cls._registry:
            raise ValueError(f"Operator {op_name} not found in registry.")
        return cls._registry[op_name]

    @classmethod
    def get_supported_ops(cls):
        """Returns a dict mapping original classes to quantized classes."""
        return cls._supported_ops

    @classmethod
    def get_supported_functions(cls):
        """Returns a set of supported functional operations."""
        return cls._supported_functions

    @classmethod
    def is_supported(cls, module_cls):
        """Checks if a module class is supported."""
        return module_cls in cls._supported_ops

    @classmethod
    def get_quantized_op(cls, original_cls):
        """Returns the quantized class for a given original class, or None if not supported."""
        return cls._supported_ops.get(original_cls)

    @classmethod
    def is_activation(cls, op_name: str):
        """Checks if the given op name corresponds to an activation layer."""
        return op_name in cls._activation_ops

    @classmethod
    def get_compliance_status(cls, op_name: str):
        """Returns the custom compliance status for an op, or None."""
        return cls._compliance_status.get(op_name)

    @classmethod
    def is_under_construction(cls, op_name: str):
        """Checks if the given op is marked as under construction."""
        return op_name in cls._under_construction_ops

    @classmethod
    def is_passthrough(cls, op_name: str):
        """Checks if the given op is a pass-through (no W/A quant of its own)."""
        return op_name in cls._passthrough_ops

    @classmethod
    def get_replacement_by_name(cls, fn_name: str):
        """Returns (Observed cls, init_from_args) for an @observer function name, or None."""
        return cls._replacements_by_name.get(fn_name)

    @classmethod
    def is_quantized(cls, op_name: str) -> bool:
        """True unless the op was registered with quantized=False (observed but
        not yet performing W/A quantization arithmetic)."""
        return op_name not in cls._unquantized_ops

# Populate standard supported functions
import torch
import torch.nn.functional as F
standard_functions = [
    F.conv2d, F.linear, F.batch_norm, F.layer_norm, F.dropout, F.softmax, F.scaled_dot_product_attention,
    torch.softmax, torch.relu, F.relu, F.relu6, F.silu, F.gelu, F.hardswish,
    # torch.matmul, torch.bmm, torch.add, torch.sub, torch.mul, torch.div, torch.cat
]
for func in standard_functions:
    OpRegistry.register_function(func)
