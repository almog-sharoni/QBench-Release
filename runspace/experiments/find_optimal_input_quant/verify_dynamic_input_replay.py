#!/usr/bin/env python3
"""Verify dynamic-input behavior by replaying exact per-batch chunk-format choices."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fix for container permission issues
os.environ.setdefault("TORCH_HOME", "/tmp/torch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from runspace.core.runner import Runner
from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (  # noqa: E402
    _build_input_quant_config,
    candidate_formats as DEFAULT_CANDIDATE_FORMATS,
)
from src.eval.metrics import MetricsEngine  # noqa: E402
from src.ops.quant_base import quantize_tensor  # noqa: E402
from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer  # noqa: E402
from src.registry.op_registry import OpRegistry  # noqa: E402


class BatchRecordingDynamicInputQuantizer(DynamicInputQuantizer):
    """Dynamic quantizer that records exact per-batch chunk-format choices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fmt_to_idx = {fmt: idx for idx, fmt in enumerate(self.candidate_formats)}
        self.current_batch_plan: Dict[str, List[bytes]] = {}

    def begin_batch(self):
        self.current_batch_plan = {}

    def consume_batch_plan(self) -> Dict[str, List[bytes]]:
        plan = self.current_batch_plan
        self.current_batch_plan = {}
        return plan

    def _get_hook(self, layer_name):
        def hook_fn(module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None

            if layer_name in self.post_relu_layers:
                candidates = self.ufp_candidates or self.non_ufp_candidates
            else:
                candidates = self.non_ufp_candidates or self.ufp_candidates

            x_quantized, chunk_formats = self._select_best_format(x, layer_name, candidates)

            # Store exact chunk choices for this layer invocation in this batch.
            encoded = bytes(self._fmt_to_idx[fmt] for fmt in chunk_formats)
            self.current_batch_plan.setdefault(layer_name, []).append(encoded)

            module.input_quantization = True
            module.input_mode = "chunk"
            module.input_chunk_size = self.chunk_size
            module.input_chunk_formats = chunk_formats
            if chunk_formats:
                module.input_q_type = chunk_formats[0]
            module.rounding = "nearest"

            with torch.no_grad():
                diff = x - x_quantized
                diff_flat = diff.reshape(-1)
                x_flat = x.reshape(-1)

                self.stats["sum_l1_err"] += diff_flat.abs().sum().item()
                self.stats["sum_mse_err"] += diff_flat.pow(2).sum().item()
                self.stats["sum_l1_norm"] += x_flat.abs().sum().item()
                self.stats["sum_l2_norm"] += x_flat.pow(2).sum().item()

            return None

        return hook_fn


class BatchReplayInputQuantizer:
    """Replay exact per-batch per-layer chunk-format choices captured earlier."""

    def __init__(self, model, chunk_size=128, candidate_formats=None):
        self.model = model
        self.chunk_size = chunk_size
        self.candidate_formats = list(candidate_formats or DEFAULT_CANDIDATE_FORMATS)
        self._idx_to_fmt = dict(enumerate(self.candidate_formats))
        self.hooks = []
        self.hooked_modules = []
        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())
        self.current_batch_plan: Dict[str, List[bytes]] = {}
        self.current_offsets: Dict[str, int] = defaultdict(int)
        self.layer_stats = {}
        self.stats = {
            "sum_l1_err": 0.0,
            "sum_mse_err": 0.0,
            "sum_l1_norm": 0.0,
            "sum_l2_norm": 0.0,
        }

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.supported_ops):
                self.hooks.append(module.register_forward_pre_hook(self._get_hook(name)))
                self.hooked_modules.append(module)

    def load_batch_plan(self, batch_plan: Dict[str, List[bytes]]):
        self.current_batch_plan = batch_plan
        self.current_offsets = defaultdict(int)

    def assert_batch_fully_consumed(self):
        leftovers = []
        for layer_name, encoded_runs in self.current_batch_plan.items():
            used = self.current_offsets.get(layer_name, 0)
            if used != len(encoded_runs):
                leftovers.append(f"{layer_name}: used {used}/{len(encoded_runs)}")
        if leftovers:
            raise RuntimeError("Replay did not consume all recorded layer invocations: " + "; ".join(leftovers[:10]))

    def _decode_chunk_formats(self, encoded: bytes) -> List[str]:
        return [self._idx_to_fmt[idx] for idx in encoded]

    def _get_hook(self, layer_name):
        def hook_fn(module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None

            layer_plan = self.current_batch_plan.get(layer_name)
            if layer_plan is None:
                raise RuntimeError(f"No replay plan found for layer '{layer_name}' in current batch.")

            offset = self.current_offsets[layer_name]
            if offset >= len(layer_plan):
                raise RuntimeError(f"Replay plan exhausted early for layer '{layer_name}'.")

            encoded = layer_plan[offset]
            self.current_offsets[layer_name] += 1
            chunk_formats = self._decode_chunk_formats(encoded)

            module.input_quantization = True
            module.input_mode = "chunk"
            module.input_chunk_size = self.chunk_size
            module.input_chunk_formats = chunk_formats
            if chunk_formats:
                module.input_q_type = chunk_formats[0]
            module.rounding = "nearest"

            with torch.no_grad():
                x_quantized, _ = quantize_tensor(
                    x,
                    q_type=chunk_formats[0] if chunk_formats else "fp32",
                    mode="chunk",
                    chunk_size=self.chunk_size,
                    rounding="nearest",
                    chunk_formats=chunk_formats,
                )
                diff = x - x_quantized
                diff_flat = diff.reshape(-1)
                x_flat = x.reshape(-1)

                self.stats["sum_l1_err"] += diff_flat.abs().sum().item()
                self.stats["sum_mse_err"] += diff_flat.pow(2).sum().item()
                self.stats["sum_l1_norm"] += x_flat.abs().sum().item()
                self.stats["sum_l2_norm"] += x_flat.pow(2).sum().item()

            counts = self.layer_stats.setdefault(layer_name, {"format_counts": {}, "type": module.__class__.__name__})
            fmt_counts = counts["format_counts"]
            for fmt in chunk_formats:
                fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1

            return None

        return hook_fn

    def get_final_stats(self):
        norm_l1 = self.stats["sum_l1_err"] / self.stats["sum_l1_norm"] if self.stats["sum_l1_norm"] > 0 else 0.0
        norm_mse = self.stats["sum_mse_err"] / self.stats["sum_l2_norm"] if self.stats["sum_l2_norm"] > 0 else 0.0
        return {
            "norm_l1": norm_l1,
            "norm_mse": norm_mse,
            "total_l1": self.stats["sum_l1_err"],
            "total_mse": self.stats["sum_mse_err"],
        }

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        for module in self.hooked_modules:
            if hasattr(module, "input_chunk_formats"):
                module.input_chunk_formats = None
        self.hooked_modules = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture dynamic input choices batch-by-batch and replay them on a fresh model."
    )
    parser.add_argument("--model_name", type=str, default="mobilevit_xxs", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--limit_batches", type=int, default=4, help="Number of batches to verify")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size")
    parser.add_argument("--metric", type=str, default="l1", help="Dynamic metric: l1 or mse")
    parser.add_argument(
        "--excluded_ops",
        type=str,
        default="LayerNorm",
        help="Comma-separated op names to exclude from quantization",
    )
    parser.add_argument(
        "--candidate_formats",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_FORMATS),
        help="Comma-separated candidate formats",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "replay_verify"),
        help="Output directory for temporary model artifacts",
    )
    args = parser.parse_args()
    args.excluded_ops = [op.strip() for op in args.excluded_ops.split(",") if op.strip()]
    args.candidate_formats = [fmt.strip() for fmt in args.candidate_formats.split(",") if fmt.strip()]
    return args


def _prepare_model_triplet(runner: Runner, args: argparse.Namespace, run_name: str):
    config = _build_input_quant_config(
        args,
        args.model_name,
        args.weights,
        "fp32",
        quantize_first_layer=False,
    )
    run_dir = os.path.join(args.output_dir, args.model_name, run_name)
    os.makedirs(run_dir, exist_ok=True)
    model, adapter, _ = runner.prepare_model_with_materialized_weights(config=config, output_dir=run_dir)
    return model, adapter, config


def _running_top1_match(outputs_a: torch.Tensor, outputs_b: torch.Tensor) -> float:
    if outputs_a.dim() != 2 or outputs_b.dim() != 2:
        return 0.0
    pred_a = outputs_a.argmax(dim=1)
    pred_b = outputs_b.argmax(dim=1)
    return pred_a.eq(pred_b).float().mean().item()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    runner = Runner(device)
    loader_cfg = _build_input_quant_config(
        args,
        args.model_name,
        args.weights,
        "fp32",
        quantize_first_layer=False,
    )
    loader = runner.setup_data_loader(loader_cfg)
    if loader is None:
        raise RuntimeError("Failed to build data loader.")

    dynamic_model = replay_model = dynamic_adapter = replay_adapter = None
    dynamic_quantizer = replay_quantizer = None

    dynamic_metrics = MetricsEngine()
    replay_metrics = MetricsEngine()
    aggregate_counts = Counter()
    total_examples = 0
    batch_top1_matches = []
    batch_max_abs_diffs = []
    batch_mean_abs_diffs = []

    try:
        dynamic_model, dynamic_adapter, _ = _prepare_model_triplet(runner, args, "dynamic_capture")
        replay_model, replay_adapter, _ = _prepare_model_triplet(runner, args, "replay_exact")

        dynamic_quantizer = BatchRecordingDynamicInputQuantizer(
            model=dynamic_model,
            metric=args.metric,
            chunk_size=args.chunk_size,
            candidate_formats=args.candidate_formats,
        )
        replay_quantizer = BatchReplayInputQuantizer(
            model=replay_model,
            chunk_size=args.chunk_size,
            candidate_formats=args.candidate_formats,
        )
        dynamic_quantizer.register_hooks()
        replay_quantizer.register_hooks()

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.limit_batches > 0 and batch_idx >= args.limit_batches:
                    break

                inputs, targets = dynamic_adapter.prepare_batch(batch)
                inputs = inputs.to(device)
                targets = targets.to(device)

                dynamic_quantizer.begin_batch()
                dynamic_outputs = dynamic_adapter.forward(dynamic_model, (inputs, targets))
                dynamic_metrics.update(dynamic_outputs, targets)
                batch_plan = dynamic_quantizer.consume_batch_plan()

                for layer_runs in batch_plan.values():
                    for encoded in layer_runs:
                        aggregate_counts.update(args.candidate_formats[idx] for idx in encoded)

                replay_quantizer.load_batch_plan(copy.deepcopy(batch_plan))
                replay_outputs = replay_adapter.forward(replay_model, (inputs, targets))
                replay_metrics.update(replay_outputs, targets)
                replay_quantizer.assert_batch_fully_consumed()

                diff = (dynamic_outputs - replay_outputs).detach()
                max_abs_diff = diff.abs().max().item()
                mean_abs_diff = diff.abs().mean().item()
                top1_match = _running_top1_match(dynamic_outputs, replay_outputs)

                batch_max_abs_diffs.append(max_abs_diff)
                batch_mean_abs_diffs.append(mean_abs_diff)
                batch_top1_matches.append(top1_match)
                total_examples += targets.size(0)

                print(
                    f"Batch {batch_idx + 1}: "
                    f"logit_max_abs_diff={max_abs_diff:.6e}, "
                    f"logit_mean_abs_diff={mean_abs_diff:.6e}, "
                    f"top1_pred_match={top1_match:.4%}"
                )

        dynamic_summary = dynamic_metrics.compute()
        replay_summary = replay_metrics.compute()
        dynamic_stats = dynamic_quantizer.get_final_stats()
        replay_stats = replay_quantizer.get_final_stats()

        print()
        print("Replay Verification Summary")
        print(f"Model: {args.model_name}")
        print(f"Metric: {args.metric}")
        print(f"Batches checked: {len(batch_top1_matches)}")
        print(f"Examples checked: {total_examples}")
        print()
        print(
            f"Dynamic acc1={dynamic_summary['acc1']:.3f}, acc5={dynamic_summary['acc5']:.3f}, "
            f"certainty={dynamic_summary['certainty']:.6f}"
        )
        print(
            f"Replay  acc1={replay_summary['acc1']:.3f}, acc5={replay_summary['acc5']:.3f}, "
            f"certainty={replay_summary['certainty']:.6f}"
        )
        print(
            f"Acc1 gap={dynamic_summary['acc1'] - replay_summary['acc1']:+.6f}, "
            f"Acc5 gap={dynamic_summary['acc5'] - replay_summary['acc5']:+.6f}"
        )
        print()
        print(
            f"Dynamic norm_l1={dynamic_stats['norm_l1']:.6e}, norm_mse={dynamic_stats['norm_mse']:.6e}"
        )
        print(
            f"Replay  norm_l1={replay_stats['norm_l1']:.6e}, norm_mse={replay_stats['norm_mse']:.6e}"
        )
        print()
        print(
            f"Average batch top1 prediction match: "
            f"{(sum(batch_top1_matches) / len(batch_top1_matches)) if batch_top1_matches else 0.0:.4%}"
        )
        print(
            f"Average batch max abs logit diff: "
            f"{(sum(batch_max_abs_diffs) / len(batch_max_abs_diffs)) if batch_max_abs_diffs else 0.0:.6e}"
        )
        print(
            f"Average batch mean abs logit diff: "
            f"{(sum(batch_mean_abs_diffs) / len(batch_mean_abs_diffs)) if batch_mean_abs_diffs else 0.0:.6e}"
        )
        print()
        print("Captured format share over verified batches:")
        total_chunks = sum(aggregate_counts.values())
        for fmt, count in aggregate_counts.most_common(10):
            share = (count / total_chunks) if total_chunks else 0.0
            print(f"  {fmt:>10}  {count:>12}  {share:.4%}")

    finally:
        if dynamic_quantizer is not None:
            dynamic_quantizer.cleanup()
        if replay_quantizer is not None:
            replay_quantizer.cleanup()
        if "loader" in locals():
            runner._shutdown_dataloader_workers(loader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
