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
    _replacements_by_name = {} # Maps upstream @observer function __name__ -> (Observed cls, init_from_args dict)
    _observed_variants = {} # replaces -> {variant_tag: (cls, init_from_args)}
    _unquantized_ops = set() # Ops registered with quantized=False — observed but no actual W/A quantization

    @classmethod
    def register(cls, op_name: str, original_cls=None, *, replaces=None, init_from_args=None,
                 is_activation=False, compliance_status=None, under_construction=False,
                 quantized=True, variant: str = None, default: bool = False):
        def decorator(cls_impl):
            cls._registry[op_name] = cls_impl
            if original_cls:
                cls._supported_ops[original_cls] = cls_impl
            if replaces is not None:
                init_args = dict(init_from_args or {})
                if variant is not None:
                    cls._observed_variants.setdefault(replaces, {})[variant] = (cls_impl, init_args)
                is_default = default or variant is None
                if is_default:
                    existing = cls._replacements_by_name.get(replaces)
                    # Two distinct class objects with the same qualified name come
                    # from the dual `src.*` / `runspace.src.*` import aliasing —
                    # treat as the same registration. Only object-distinct, name-
                    # distinct classes are a real conflict.
                    if existing is not None and existing[0] is not cls_impl \
                            and getattr(existing[0], "__qualname__", existing[0].__name__) \
                                != getattr(cls_impl, "__qualname__", cls_impl.__name__):
                        raise ValueError(
                            f"Multiple default replacements registered for '{replaces}': "
                            f"{existing[0].__name__} and {cls_impl.__name__}. "
                            f"Only one variant may use default=True (or omit `variant`)."
                        )
                    cls._replacements_by_name[replaces] = (cls_impl, init_args)
            if is_activation:
                cls._activation_ops.add(op_name)
            if compliance_status:
                cls._compliance_status[op_name] = compliance_status
            if under_construction:
                cls._under_construction_ops.add(op_name)
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
    def get_replacement_by_name(cls, fn_name: str):
        """Returns (Observed cls, init_from_args) for an @observer function name, or None."""
        return cls._replacements_by_name.get(fn_name)

    @classmethod
    def get_observed_variant(cls, fn_name: str, variant: str):
        """Returns (Observed cls, init_from_args) for a tagged variant of an
        @observer function, or None when the tag is unknown."""
        return cls._observed_variants.get(fn_name, {}).get(variant)

    @classmethod
    def list_observed_variants(cls, fn_name: str):
        """Returns the list of variant tags registered against an @observer
        function (empty when none were tagged)."""
        return list(cls._observed_variants.get(fn_name, {}).keys())

    @classmethod
    def resolve_observed_class_from_config(cls, fn_name: str, cfg: dict | None,
                                           parent_path: str = ""):
        """Resolve the variant class declared for ``fn_name`` in the active
        config's ``layers:`` block, or the default class when no variant is
        set. Mirrors the lookup in ``observer._resolve_observed_entry`` so
        report-side code can pick the same class the dispatch used.

        Lookup keys on ``layers``: qualified path (``<parent>.<fn>``), bare
        function name, then any key ending in ``.<fn>``.
        """
        layers = (cfg or {}).get("layers", {}) if isinstance(cfg, dict) else {}
        layer_cfg = None
        if layers:
            qualified = f"{parent_path}.{fn_name}" if parent_path else fn_name
            layer_cfg = layers.get(qualified) or layers.get(fn_name)
            if layer_cfg is None:
                suffix = f".{fn_name}"
                for key, val in layers.items():
                    if isinstance(key, str) and key.endswith(suffix):
                        layer_cfg = val
                        break
        variant_tag = layer_cfg.get("variant") if isinstance(layer_cfg, dict) else None
        if variant_tag is not None:
            entry = cls.get_observed_variant(fn_name, variant_tag)
            if entry is not None:
                return entry[0]
        default = cls.get_replacement_by_name(fn_name)
        return default[0] if default is not None else None

    @classmethod
    def iter_observed_classes(cls, fn_name: str):
        """Yields every class registered as the default replacement OR any
        variant of the given @observer function name, default first.
        De-duplicated by class object identity."""
        seen = set()
        default = cls._replacements_by_name.get(fn_name)
        if default is not None and id(default[0]) not in seen:
            seen.add(id(default[0]))
            yield default[0]
        for entry in cls._observed_variants.get(fn_name, {}).values():
            if id(entry[0]) not in seen:
                seen.add(id(entry[0]))
                yield entry[0]

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
