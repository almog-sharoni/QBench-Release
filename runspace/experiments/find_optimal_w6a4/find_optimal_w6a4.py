
import os
import sys
import gc
import json
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.src.quantization.quantizer import quantize
from runspace.src.ops.quant_base import calculate_scale
from runspace.src.database.handler import RunDatabase
from runspace.src.eval.metrics import compute_certainty



# ---------------------------------------------------------------------------
# Default format lists
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_FORMATS_6BIT = [
     'fp6_e1m4', 'fp6_e2m3', 'fp6_e3m2', 'fp6_e4m1', 'fp6_e5m0',
]

DEFAULT_ACTIVATION_FORMATS_4BIT = [
     'fp4_e1m2', 'fp4_e2m1', 'fp4_e3m0',
]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "W6A4 experiment: sweep 6-bit weight formats × 4-bit activation formats "
            "to test whether independent best = combined best."
        )
    )
    # Model
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--model_source", type=str, default="torchvision",
                        help="Model source library: 'torchvision' or 'timm'")
    parser.add_argument("--models_file", type=str, default=None,
                        help="YAML file with list of models (runs all models sequentially)")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="classification")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit_batches", type=int, default=-1,
                        help="Limit batches per run (-1 = all)")

    # Format lists (comma-separated)
    parser.add_argument(
        "--weight_formats", type=str,
        default=','.join(DEFAULT_WEIGHT_FORMATS_6BIT),
        help="Comma-separated 6-bit weight formats to sweep"
    )
    parser.add_argument(
        "--activation_formats", type=str,
        default=','.join(DEFAULT_ACTIVATION_FORMATS_4BIT),
        help="Comma-separated 4-bit activation formats to sweep"
    )

    # Quantization settings
    parser.add_argument("--chunk_size", type=int, default=128,
                        help="Chunk size for both weight and activation quantization")

    # Phase control
    parser.add_argument("--skip_phase_a", action="store_true",
                        help="Skip Phase A (weight-only sweep)")
    parser.add_argument("--skip_phase_b", action="store_true",
                        help="Skip Phase B (activation-only sweep)")
    parser.add_argument("--skip_phase_c", action="store_true",
                        help="Skip Phase C (cross sweep)")

    # Misc
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--force_rerun", action="store_true",
                        help="Re-run runs even if already in DB")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Weight quantization — same adapter config as weight_quant_baseline
# ---------------------------------------------------------------------------

def _make_weight_quant_config(args, fmt, chunk_size):
    """Adapter config identical to weight_quant_baseline."""
    return {
        'model': {'name': args.model_name, 'weights': args.weights, 'source': args.model_source},
        'adapter': {
            'type': 'generic',
            'quantized_ops': ['-1'],
            'input_quantization': False,
        },
        'quantization': {
            'format': fmt,
            'weight_mode': 'chunk',
            'weight_chunk_size': chunk_size,
        },
        'dataset': {
            'type': args.dataset_type,
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }


def _get_or_build_quantized_state_dict(args, fmt, chunk_size, weights_dir, force_rebuild=False):
    """
    Build (and cache on disk) the calibrated model state dict using the exact
    same adapter config as weight_quant_baseline. The saved file can be loaded
    back into an identical model structure to reproduce the same forward pass.
    """
    os.makedirs(weights_dir, exist_ok=True)
    cache_path = os.path.join(weights_dir, f"{fmt}_chunk{chunk_size}.pt")

    if not force_rebuild and os.path.exists(cache_path):
        print(f"  [weights] Loading cached: {cache_path}")
        return cache_path

    print(f"  [weights] Building quantized model (weight_quant_baseline config) → {cache_path}")
    config = _make_weight_quant_config(args, fmt, chunk_size)
    adapter = create_adapter(config)
    clean_sd = {k: v.clone().contiguous() for k, v in adapter.model.state_dict().items()}
    torch.save(clean_sd, cache_path)
    print(f"  [weights] Saved: {cache_path}")

    del adapter
    gc.collect()
    torch.cuda.empty_cache()

    return cache_path


def _load_model_with_uniform_weights(args, device, fmt, chunk_size, weights_dir, force_rebuild=False):
    """
    Load a weight-quantized model using the same adapter structure as
    weight_quant_baseline, restoring weights from the cached state dict.
    """
    cache_path = _get_or_build_quantized_state_dict(
        args, fmt, chunk_size, weights_dir, force_rebuild=force_rebuild
    )
    config = _make_weight_quant_config(args, fmt, chunk_size)
    config['adapter']['skip_calibration'] = True  # weights restored from cache below
    adapter = create_adapter(config)
    state_dict = torch.load(cache_path, map_location='cpu')
    # Buffers registered as None (weight_scale, weight_fp8) are absent from
    # model.state_dict(), causing strict=True to fail with "unexpected keys".
    # Pre-initialise them as empty tensors matching the cached shapes so the
    # strict load can proceed and assign=True will overwrite them correctly.
    for fqn, tensor in state_dict.items():
        parts = fqn.split('.')
        mod = adapter.model
        try:
            for part in parts[:-1]:
                mod = getattr(mod, part)
            buf_name = parts[-1]
            if buf_name in mod._buffers and mod._buffers[buf_name] is None:
                mod.register_buffer(buf_name, torch.empty_like(tensor))
        except AttributeError:
            pass
    # assign=True replaces buffers/parameters instead of copy_()-ing into them,
    # avoiding the overlapping-memory error from stride-0 scale tensors.
    try:
        adapter.model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        # PyTorch < 2.1 fallback: contiguify model buffers before loading
        for module in adapter.model.modules():
            for name, buf in list(module._buffers.items()):
                if buf is not None and not buf.is_contiguous():
                    module._buffers[name] = buf.contiguous()
        adapter.model.load_state_dict(state_dict, strict=True)
    adapter.model.to(device).eval()
    return adapter.model, adapter


def _load_fp32_model(args, device):
    """Load model with original fp32 weights."""
    config = _make_config(args)
    adapter = create_adapter(config)
    model = adapter.model.to(device).eval()
    return model, adapter


# ---------------------------------------------------------------------------
# Uniform activation quantization (hook-based)
# ---------------------------------------------------------------------------

def _quantize_tensor(x, fmt, chunk_size):
    """Chunk-wise quantization of a tensor (shared by Phase B and Phase C hooks)."""
    original_shape = x.shape
    flat = x.reshape(-1)
    pad_len = 0
    if flat.numel() % chunk_size != 0:
        pad_len = chunk_size - (flat.numel() % chunk_size)
        flat = F.pad(flat, (0, pad_len))
    chunks = flat.view(-1, chunk_size)
    scale = calculate_scale(chunks.abs().amax(dim=1, keepdim=True).clamp(min=1e-5), fmt)
    q_chunks = quantize(chunks / scale, q_type=fmt, validate=False) * scale
    flat_q = q_chunks.reshape(-1)
    if pad_len > 0:
        flat_q = flat_q[:-pad_len]
    return flat_q.reshape(original_shape)


class UniformInputQuantizer:
    """
    Registers forward pre-hooks on all Conv2d / Linear layers (and any
    registered quantized ops) to quantize activations uniformly to `fmt`.
    """
    def __init__(self, model, fmt, chunk_size=128):
        self.model = model
        self.fmt = fmt
        self.chunk_size = chunk_size
        self.hooks = []
        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())
        self.stats = {
            'sum_l1_err': 0.0, 'sum_mse_err': 0.0,
            'sum_l1_norm': 0.0, 'sum_l2_norm': 0.0,
        }

    def _quantize(self, x):
        return _quantize_tensor(x, self.fmt, self.chunk_size)

    def _make_hook(self):
        def hook(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            x = inputs[0]
            x_q = self._quantize(x)
            with torch.no_grad():
                diff = x - x_q
                self.stats['sum_l1_err']  += diff.abs().sum().item()
                self.stats['sum_mse_err'] += diff.pow(2).sum().item()
                self.stats['sum_l1_norm'] += x.abs().sum().item()
                self.stats['sum_l2_norm'] += x.pow(2).sum().item()
            return (x_q,) + inputs[1:]
        return hook

    def register_hooks(self):
        hook_fn = self._make_hook()
        for name, module in self.model.named_modules():
            if isinstance(module, self.supported_ops) or isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_pre_hook(hook_fn))

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_stats(self):
        l1_norm  = self.stats['sum_l1_norm']
        l2_norm  = self.stats['sum_l2_norm']
        return {
            'norm_l1':  self.stats['sum_l1_err']  / l1_norm  if l1_norm  > 0 else 0.0,
            'norm_mse': self.stats['sum_mse_err'] / l2_norm if l2_norm > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _make_config(args):
    return {
        'model': {'name': args.model_name, 'weights': args.weights, 'source': args.model_source},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'type': args.dataset_type,
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }


def _build_loader(args, device):
    runner = Runner(device)
    loader = runner.setup_data_loader(_make_config(args))
    return loader


def _run_inference(model, adapter, loader, device, args,
                   act_quantizer=None, input_fmt=None, chunk_size=128, desc=""):
    """
    Run inference. Two quantization modes (mutually exclusive):
      - input_fmt: quantize only the input image batch chunk-wise before the forward
        pass (matches input_quant_baseline behaviour).
      - act_quantizer: hook-based quantization at every layer input (Phase C).
    """
    correct1 = correct5 = total = 0
    total_certainty = 0.0
    batch_count = 0
    limit = args.limit_batches if args.limit_batches > 0 else float('inf')
    input_stats = {'sum_l1_err': 0.0, 'sum_mse_err': 0.0,
                   'sum_l1_norm': 0.0, 'sum_l2_norm': 0.0}

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch_count >= limit:
                break
            images, labels = adapter.prepare_batch(batch)
            images, labels = images.to(device), labels.to(device)
            if input_fmt is not None:
                images_q = _quantize_tensor(images, input_fmt, chunk_size)
                diff = images - images_q
                input_stats['sum_l1_err']  += diff.abs().sum().item()
                input_stats['sum_mse_err'] += diff.pow(2).sum().item()
                input_stats['sum_l1_norm'] += images.abs().sum().item()
                input_stats['sum_l2_norm'] += images.pow(2).sum().item()
                images = images_q
            outputs = model(images)
            total_certainty += compute_certainty(outputs) * images.size(0)
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            correct1 += correct[:1].reshape(-1).float().sum().item()
            correct5 += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)
            batch_count += 1

    acc1 = 100.0 * correct1 / total if total > 0 else 0.0
    acc5 = 100.0 * correct5 / total if total > 0 else 0.0
    certainty = total_certainty / total if total > 0 else 0.0
    if act_quantizer is not None:
        act_stats = act_quantizer.get_stats()
    elif input_fmt is not None:
        l1_norm = input_stats['sum_l1_norm']
        l2_norm = input_stats['sum_l2_norm']
        act_stats = {
            'norm_l1':  input_stats['sum_l1_err']  / l1_norm if l1_norm > 0 else 0.0,
            'norm_mse': input_stats['sum_mse_err'] / l2_norm if l2_norm > 0 else 0.0,
        }
    else:
        act_stats = None
    return acc1, acc5, certainty, act_stats


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _run_exists(db, model_name, experiment_type, weight_dt, activation_dt):
    runs = db.get_runs()
    if runs.empty:
        return False
    match = runs[
        (runs['model_name']      == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['weight_dt']       == weight_dt) &
        (runs['activation_dt']   == activation_dt) &
        (runs['status']          == 'SUCCESS')
    ]
    return not match.empty


def _get_or_run_fp32_ref(args, device, db, model_name):
    runs = db.get_runs()
    if not args.force_rerun and not runs.empty:
        fp32_rows = runs[
            (runs['model_name']    == model_name) &
            (runs['weight_dt']     == 'fp32') &
            (runs['activation_dt'] == 'fp32') &
            (runs['status']        == 'SUCCESS') &
            (runs['acc1'].notna())
        ]
        if not fp32_rows.empty:
            row = fp32_rows.iloc[0]
            ref_acc1 = float(row['acc1'])
            ref_acc5 = float(row['acc5'])
            ref_cert = float(row['certainty']) if row['certainty'] is not None else 0.0
            print(f"[FP32 ref] Found in DB: Top1={ref_acc1:.2f}%, Top5={ref_acc5:.2f}%")
            return ref_acc1, ref_acc5, ref_cert

    print("[FP32 ref] Not in DB — running inference ...")
    model, adapter = _load_fp32_model(args, device)
    loader = _build_loader(args, device)
    acc1, acc5, cert, _ = _run_inference(model, adapter, loader, device, args, desc="FP32 ref")
    del model, adapter
    gc.collect(); torch.cuda.empty_cache()
    print(f"[FP32 ref] Top1={acc1:.2f}%, Top5={acc5:.2f}%")
    db.log_run(
        model_name=model_name, weight_dt="fp32", activation_dt="fp32",
        task_type=args.dataset_type,
        acc1=acc1, acc5=acc5, ref_acc1=acc1, ref_acc5=acc5, ref_certainty=cert,
        experiment_type="fp32_ref", status="SUCCESS", certainty=cert,
        config_json=_serialize_config(_make_config(args)),
    )
    return acc1, acc5, cert


def _serialize_config(cfg, activation_fmt=None, chunk_size=128, weights_source=None):
    """
    Serialize a full adapter config dict to JSON for DB storage.
    Optionally merges activation quantization fields (Phase B/C).
    weights_source: human-readable string describing where weights came from.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    # Remove runtime-only flag — not meaningful as a stored config field
    cfg.get('adapter', {}).pop('skip_calibration', None)
    if weights_source is not None:
        cfg.setdefault('quantization', {})
        cfg['quantization']['weights_source'] = weights_source
    if activation_fmt is not None:
        cfg.setdefault('quantization', {})
        cfg['quantization']['input_quantization'] = True
        cfg['quantization']['activation_format'] = activation_fmt
        cfg['quantization']['activation_mode'] = 'chunk'
        cfg['quantization']['activation_chunk_size'] = chunk_size
    return json.dumps(cfg)


def _best_from_phase(db, model_name, experiment_type):
    """Return (weight_dt, activation_dt, acc1) of the top run in the given experiment phase."""
    runs = db.get_runs()
    if runs.empty:
        return None, None, None
    phase_runs = runs[
        (runs['model_name']      == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['status']          == 'SUCCESS') &
        (runs['acc1'].notna())
    ]
    if phase_runs.empty:
        return None, None, None
    best = phase_runs.loc[phase_runs['acc1'].idxmax()]
    return best['weight_dt'], best['activation_dt'], float(best['acc1'])


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def process_single_model(args, device):
    model_name = args.model_name
    weight_formats    = [f.strip() for f in args.weight_formats.split(',')]
    activation_formats = [f.strip() for f in args.activation_formats.split(',')]

    db = RunDatabase()
    weights_dir = os.path.join(args.output_dir, model_name, "weights")

    print(f"\n{'='*60}")
    print(f" W6A4 EXPERIMENT: {model_name}")
    print(f"  Weight formats (6-bit)     : {weight_formats}")
    print(f"  Activation formats (4-bit) : {activation_formats}")
    print(f"{'='*60}")

    # FP32 reference
    ref_acc1, ref_acc5, ref_cert = _get_or_run_fp32_ref(args, device, db, model_name)

    # -----------------------------------------------------------------------
    # Phase A — weight-only sweep (quantized weights × fp32 activations)
    # -----------------------------------------------------------------------
    if not args.skip_phase_a:
        print(f"\n{'─'*60}")
        print(f" Phase A: 6-bit weight sweep × fp32 activations")
        print(f"{'─'*60}")

        loader = _build_loader(args, device)

        for w_fmt in weight_formats:
            if not args.force_rerun and _run_exists(
                db, model_name, "w6a4_weight_only", w_fmt, "fp32"
            ):
                print(f"  [A] {w_fmt} × fp32 — already in DB, skipping.")
                continue

            print(f"  [A] {w_fmt} × fp32 ...")
            model, adapter = _load_model_with_uniform_weights(
                args, device, w_fmt, args.chunk_size, weights_dir,
                force_rebuild=args.force_rerun,
            )
            try:
                acc1, acc5, cert, _ = _run_inference(
                    model, adapter, loader, device, args,
                    desc=f"A: {w_fmt}"
                )
            except Exception as e:
                print(f"  [A] ERROR ({w_fmt}): {e}")
                db.log_run(
                    model_name=model_name, weight_dt=w_fmt, activation_dt="fp32",
                    task_type=args.dataset_type,
                    acc1=0.0, acc5=0.0, ref_acc1=ref_acc1, ref_acc5=ref_acc5,
                    ref_certainty=ref_cert, experiment_type="w6a4_weight_only",
                    status="ERROR",
                    config_json=_serialize_config(
                        _make_weight_quant_config(args, w_fmt, args.chunk_size),
                        weights_source="cached_pre_quantized",
                    ),
                )
                del model, adapter; gc.collect(); torch.cuda.empty_cache()
                continue

            print(f"      Top1={acc1:.2f}%, Top5={acc5:.2f}%")
            db.log_run(
                model_name=model_name, weight_dt=w_fmt, activation_dt="fp32",
                task_type=args.dataset_type,
                acc1=acc1, acc5=acc5, ref_acc1=ref_acc1, ref_acc5=ref_acc5,
                ref_certainty=ref_cert, experiment_type="w6a4_weight_only",
                status="SUCCESS", certainty=cert,
                config_json=_serialize_config(
                    _make_weight_quant_config(args, w_fmt, args.chunk_size),
                    weights_source="cached_pre_quantized",
                ),
            )
            del model, adapter; gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase B — activation-only sweep (fp32 weights × quantized activations)
    # -----------------------------------------------------------------------
    if not args.skip_phase_b:
        print(f"\n{'─'*60}")
        print(f" Phase B: fp32 weights × 4-bit activation sweep")
        print(f"{'─'*60}")

        loader = _build_loader(args, device)

        for a_fmt in activation_formats:
            if not args.force_rerun and _run_exists(
                db, model_name, "w6a4_input_only", "fp32", a_fmt
            ):
                print(f"  [B] fp32 × {a_fmt} — already in DB, skipping.")
                continue

            print(f"  [B] fp32 × {a_fmt} ...")
            model, adapter = _load_fp32_model(args, device)
            try:
                acc1, acc5, cert, act_stats = _run_inference(
                    model, adapter, loader, device, args,
                    input_fmt=a_fmt, chunk_size=args.chunk_size, desc=f"B: {a_fmt}"
                )
            except Exception as e:
                print(f"  [B] ERROR ({a_fmt}): {e}")
                db.log_run(
                    model_name=model_name, weight_dt="fp32", activation_dt=a_fmt,
                    task_type=args.dataset_type,
                    acc1=0.0, acc5=0.0, ref_acc1=ref_acc1, ref_acc5=ref_acc5,
                    ref_certainty=ref_cert, experiment_type="w6a4_input_only",
                    status="ERROR",
                    config_json=_serialize_config(_make_config(args), activation_fmt=a_fmt, chunk_size=args.chunk_size),
                )
                del model, adapter; gc.collect(); torch.cuda.empty_cache()
                continue

            norm_l1  = act_stats['norm_l1']  if act_stats else None
            norm_mse = act_stats['norm_mse'] if act_stats else None
            print(
                f"      Top1={acc1:.2f}%, Top5={acc5:.2f}%"
                + (f", NormL1={norm_l1:.4e}, NormMSE={norm_mse:.4e}" if norm_l1 is not None else "")
            )
            db.log_run(
                model_name=model_name, weight_dt="fp32", activation_dt=a_fmt,
                task_type=args.dataset_type,
                acc1=acc1, acc5=acc5, ref_acc1=ref_acc1, ref_acc5=ref_acc5,
                ref_certainty=ref_cert, experiment_type="w6a4_input_only",
                status="SUCCESS", certainty=cert,
                mse=norm_mse, l1=norm_l1,
                config_json=_serialize_config(_make_config(args), activation_fmt=a_fmt, chunk_size=args.chunk_size),
            )
            del model, adapter; gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase C — cross sweep (all weight × all activation combinations)
    # -----------------------------------------------------------------------
    if not args.skip_phase_c:
        print(f"\n{'─'*60}")
        print(f" Phase C: 6-bit weight × 4-bit activation cross sweep")
        print(f" ({len(weight_formats)} × {len(activation_formats)} = "
              f"{len(weight_formats) * len(activation_formats)} combos)")
        print(f"{'─'*60}")

        loader = _build_loader(args, device)

        for w_fmt in weight_formats:
            for a_fmt in activation_formats:
                if not args.force_rerun and _run_exists(
                    db, model_name, "w6a4_cross", w_fmt, a_fmt
                ):
                    print(f"  [C] {w_fmt} × {a_fmt} — already in DB, skipping.")
                    continue

                print(f"  [C] {w_fmt} × {a_fmt} ...")
                model, adapter = _load_model_with_uniform_weights(
                    args, device, w_fmt, args.chunk_size, weights_dir,
                    force_rebuild=False,  # weights_dir already built in Phase A
                )
                act_q = UniformInputQuantizer(model, fmt=a_fmt, chunk_size=args.chunk_size)
                act_q.register_hooks()
                try:
                    acc1, acc5, cert, act_stats = _run_inference(
                        model, adapter, loader, device, args,
                        act_quantizer=act_q, desc=f"C: {w_fmt}×{a_fmt}"
                    )
                except Exception as e:
                    print(f"  [C] ERROR ({w_fmt}×{a_fmt}): {e}")
                    act_q.cleanup()
                    db.log_run(
                        model_name=model_name, weight_dt=w_fmt, activation_dt=a_fmt,
                        task_type=args.dataset_type,
                        acc1=0.0, acc5=0.0, ref_acc1=ref_acc1, ref_acc5=ref_acc5,
                        ref_certainty=ref_cert, experiment_type="w6a4_cross",
                        status="ERROR",
                        config_json=_serialize_config(
                            _make_weight_quant_config(args, w_fmt, args.chunk_size),
                            activation_fmt=a_fmt, chunk_size=args.chunk_size,
                            weights_source="cached_pre_quantized",
                        ),
                    )
                    del model, adapter; gc.collect(); torch.cuda.empty_cache()
                    continue
                finally:
                    act_q.cleanup()

                norm_l1  = act_stats['norm_l1']  if act_stats else None
                norm_mse = act_stats['norm_mse'] if act_stats else None
                print(
                    f"      Top1={acc1:.2f}%, Top5={acc5:.2f}%"
                    + (f", NormL1={norm_l1:.4e}, NormMSE={norm_mse:.4e}" if norm_l1 is not None else "")
                )
                db.log_run(
                    model_name=model_name, weight_dt=w_fmt, activation_dt=a_fmt,
                    task_type=args.dataset_type,
                    acc1=acc1, acc5=acc5, ref_acc1=ref_acc1, ref_acc5=ref_acc5,
                    ref_certainty=ref_cert, experiment_type="w6a4_cross",
                    status="SUCCESS", certainty=cert,
                    mse=norm_mse, l1=norm_l1,
                    config_json=_serialize_config(
                        _make_weight_quant_config(args, w_fmt, args.chunk_size),
                        activation_fmt=a_fmt, chunk_size=args.chunk_size,
                        weights_source="cached_pre_quantized",
                    ),
                )
                del model, adapter; gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary — best(A) + best(B) vs best(C)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" SUMMARY: {model_name}")
    print(f"{'='*60}")

    bw_a, _, best_a_acc = _best_from_phase(db, model_name, "w6a4_weight_only")
    _, ba_b, best_b_acc = _best_from_phase(db, model_name, "w6a4_input_only")
    bw_c, ba_c, best_c_acc = _best_from_phase(db, model_name, "w6a4_cross")

    print(f"  Phase A best weight  : {bw_a}  (Top1={best_a_acc:.2f}%)" if best_a_acc is not None
          else "  Phase A: no results")
    print(f"  Phase B best input   : {ba_b}  (Top1={best_b_acc:.2f}%)" if best_b_acc is not None
          else "  Phase B: no results")
    print(f"  Phase C best combo   : {bw_c} × {ba_c}  (Top1={best_c_acc:.2f}%)" if best_c_acc is not None
          else "  Phase C: no results")

    if best_a_acc is not None and best_b_acc is not None and best_c_acc is not None:
        # Check if the independent best combo was the winner in Phase C too
        runs = db.get_runs()
        combined_c = runs[
            (runs['model_name']      == model_name) &
            (runs['experiment_type'] == "w6a4_cross") &
            (runs['weight_dt']       == bw_a) &
            (runs['activation_dt']   == ba_b) &
            (runs['status']          == 'SUCCESS')
        ]
        if not combined_c.empty:
            ind_acc = float(combined_c.iloc[0]['acc1'])
            gap = ind_acc - best_c_acc
            print(f"\n  Independent best ({bw_a} × {ba_b})  in Phase C: Top1={ind_acc:.2f}%")
            print(f"  Gap vs global cross best: {gap:+.2f}% "
                  f"({'✓ separable' if abs(gap) < 0.1 else '✗ not separable'})")
        else:
            print(f"\n  [!] Independent best combo ({bw_a} × {ba_b}) was not run in Phase C.")
            print(f"      Re-run without --skip_phase_c (or with --force_rerun) to fill the gap.")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.models_file:
        with open(args.models_file) as f:
            models_cfg = yaml.safe_load(f)
        model_list = models_cfg if isinstance(models_cfg, list) else models_cfg.get("models", [])
        for entry in model_list:
            if isinstance(entry, dict):
                args.model_name = entry.get("name", args.model_name)
                args.weights = entry.get("weights", "DEFAULT")
                args.model_source = entry.get("source", "torchvision")
            else:
                args.model_name = str(entry)
                args.weights = "DEFAULT"
                args.model_source = "torchvision"
            process_single_model(args, device)
    else:
        process_single_model(args, device)


if __name__ == "__main__":
    main()
