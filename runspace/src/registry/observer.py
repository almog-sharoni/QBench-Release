"""@observer decorator and quantized-dispatch state.

Marks a function or class as observable. Functions are wrapped: at call time
the wrapper checks whether quantized dispatch is active and looks up an
Observed* class registered under the original function's `__name__`. If
present, the wrapper instantiates (and caches) the Observed class with init
kwargs derived from the call args via the binding declared at registration,
plus quantization kwargs from the active dispatch config, and forwards the
remaining call args to the instance. Otherwise the original body runs
unchanged. Classes are returned untouched (pure marker).
"""
from __future__ import annotations

import functools
import inspect
import sys
import threading
from collections import Counter
from contextlib import contextmanager

if __name__ == "src.registry.observer":
    sys.modules.setdefault("runspace.src.registry.observer", sys.modules[__name__])
elif __name__ == "runspace.src.registry.observer":
    sys.modules.setdefault("src.registry.observer", sys.modules[__name__])

from runspace.src.registry.op_registry import OpRegistry

_state = threading.local()
_cache: dict = {}
_signature_cache: dict = {}


def _flag() -> bool:
    return getattr(_state, "active", False)


def is_quantized_dispatch_active() -> bool:
    return _flag()


def get_active_config():
    return getattr(_state, "config", None)


@contextmanager
def quantized_dispatch(config=None, call_log=None):
    prev_active = _flag()
    prev_cfg = getattr(_state, "config", None)
    prev_log = getattr(_state, "call_log", None)
    prev_frozen = getattr(_state, "call_log_frozen", False)
    _state.active = True
    _state.config = config
    _state.call_log = call_log
    _state.call_log_frozen = False
    try:
        yield
    finally:
        _state.active = prev_active
        _state.config = prev_cfg
        _state.call_log = prev_log
        _state.call_log_frozen = prev_frozen


def _push_event(name, module):
    log = getattr(_state, "call_log", None)
    if log is None or getattr(_state, "call_log_frozen", False):
        return
    log.append((name, module))


def freeze_call_log() -> None:
    """Stop further events from being appended to the active call_log."""
    _state.call_log_frozen = True


def _current_parent_path() -> str:
    stack = getattr(_state, "module_path_stack", None)
    return stack[-1] if stack else ""


def push_module_path(path: str) -> None:
    stack = getattr(_state, "module_path_stack", None)
    if stack is None:
        stack = []
        _state.module_path_stack = stack
    stack.append(path)


def pop_module_path() -> None:
    stack = getattr(_state, "module_path_stack", None)
    if stack:
        stack.pop()


def _counts() -> Counter:
    c = getattr(_state, "counts", None)
    if c is None:
        c = Counter()
        _state.counts = c
    return c


def get_dispatch_counts() -> Counter:
    return Counter(_counts())


def reset_dispatch_counts() -> None:
    _state.counts = Counter()


def _signature(func):
    sig = _signature_cache.get(func)
    if sig is None:
        sig = inspect.signature(func)
        _signature_cache[func] = sig
    return sig


def _split_call(target, args, kwargs, init_from_args):
    """Bind args/kwargs against target's signature, peel out init_from_args
    keys, return (init_kwargs, fwd_args, fwd_kwargs)."""
    sig = _signature(target)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    init_kwargs = {}
    for init_kw, call_kw in (init_from_args or {}).items():
        if call_kw in bound.arguments:
            init_kwargs[init_kw] = bound.arguments.pop(call_kw)
    fwd_args = []
    fwd_kwargs = {}
    for name, value in bound.arguments.items():
        param = sig.parameters[name]
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
            fwd_args.append(value)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            fwd_args.extend(value)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            fwd_kwargs.update(value)
        else:
            fwd_kwargs[name] = value
    return init_kwargs, fwd_args, fwd_kwargs


def _quant_init_kwargs(cls, cfg):
    """Inspect cls.__init__ and return only the quantization kwargs it accepts."""
    if cfg is None:
        return {}
    candidates = {
        "q_type": cfg.get("format"),
        "quant_mode": cfg.get("mode"),
        "chunk_size": cfg.get("chunk_size"),
    }
    sig = _signature(cls.__init__)
    return {k: v for k, v in candidates.items() if k in sig.parameters and v is not None}


def _set_input_quantization(module):
    """Recursively flip input_quantization=True on module and any submodules
    that own that attribute."""
    if hasattr(module, "input_quantization"):
        module.input_quantization = True
    for child in module.children():
        _set_input_quantization(child)


def _ensure_capture_hooks(inst, suffix: str):
    """Register a single forward hook on the top-level cached singleton (not
    on its submodules) so each invocation pushes one event labelled with the
    upstream @observer function name into the active call_log when one is set.
    Idempotent."""
    if getattr(inst, "_qbench_capture_hooked", False):
        return
    def hook(module, inputs, output):
        parent = _current_parent_path()
        name = f"{parent}.{suffix}" if parent else suffix
        _push_event(name, module)
    inst.register_forward_hook(hook)
    inst._qbench_capture_hooked = True


def _get_or_build(cls, init_kwargs, suffix: str):
    cfg = get_active_config()
    quant_kwargs = _quant_init_kwargs(cls, cfg)
    full_kwargs = {**quant_kwargs, **init_kwargs}
    try:
        key = (cls, frozenset(full_kwargs.items()))
    except TypeError:
        key = (cls, tuple(sorted((k, repr(v)) for k, v in full_kwargs.items())))
    inst = _cache.get(key)
    if inst is None:
        inst = cls(**full_kwargs)
        if cfg is not None:
            _set_input_quantization(inst)
        inst.eval()
        _ensure_capture_hooks(inst, suffix)
        _cache[key] = inst
    return inst


def _fx_mark_opaque_in_caller(fn_name: str) -> None:
    # Register fn_name as a leaf function in the caller-module's globals so
    # torch.fx.symbolic_trace emits one call_function node instead of tracing
    # through. torch.fx.wrap itself refuses to run outside module top-level,
    # so we poke its private table directly.
    try:
        from torch.fx._symbolic_trace import _wrapped_fns_to_patch
    except Exception:
        return
    frame = inspect.stack()[2].frame
    caller_globals = frame.f_globals
    _wrapped_fns_to_patch[(id(caller_globals), fn_name)] = caller_globals


def observer(target):
    if inspect.isclass(target):
        target.__qbench_observed_class__ = True
        return target

    @functools.wraps(target)
    def wrapped(*args, **kwargs):
        if _flag():
            entry = OpRegistry.get_replacement_by_name(target.__name__)
            if entry is not None:
                cls, init_from_args = entry
                init_kwargs, fwd_args, fwd_kwargs = _split_call(target, args, kwargs, init_from_args)
                inst = _get_or_build(cls, init_kwargs, target.__name__)
                _counts()[target.__name__] += 1
                return inst(*fwd_args, **fwd_kwargs)
        return target(*args, **kwargs)

    wrapped.__wrapped__ = target
    wrapped.__qbench_observed__ = True
    _fx_mark_opaque_in_caller(target.__name__)
    return wrapped
