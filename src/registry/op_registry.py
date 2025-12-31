import torch.nn as nn

class OpRegistry:
    """
    A central mapping from logical ops to implementations.
    """
    _registry = {}
    _supported_ops = {} # Mapping from original_cls -> quantized_cls
    _activation_ops = set() # Set of quantized op names that are activations
    _compliance_status = {} # Mapping from op_name -> status message (if custom)
    _supported_functions = set() # Set of supported functional operations (e.g. F.conv2d)

    @classmethod
    def register(cls, op_name: str, original_cls=None, is_activation=False, compliance_status=None):
        def decorator(cls_impl):
            cls._registry[op_name] = cls_impl
            if original_cls:
                cls._supported_ops[original_cls] = cls_impl
            if is_activation:
                cls._activation_ops.add(op_name)
            if compliance_status:
                cls._compliance_status[op_name] = compliance_status
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

# Populate standard supported functions
import torch
import torch.nn.functional as F
standard_functions = [
    F.conv2d, F.linear, F.batch_norm, F.layer_norm, F.dropout, F.softmax, F.scaled_dot_product_attention,
    torch.softmax, torch.relu, F.relu, F.relu6, F.silu, F.gelu, F.hardswish
]
for func in standard_functions:
    OpRegistry.register_function(func)
