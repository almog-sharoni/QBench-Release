#!/usr/bin/env python3
"""Verify dynamic-input behavior by replaying exact per-batch chunk-format choices."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fix for container permission issues
os.environ.setdefault("TORCH_HOME", "/tmp/torch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from runspace.core.runner import Runner  # noqa: E402
from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (  # noqa: E402
    _build_input_quant_config,
    candidate_formats as DEFAULT_CANDIDATE_FORMATS,
)
from src.eval.metrics import MetricsEngine  # noqa: E402
from src.quantization.activation_transport import (  # noqa: E402
    ActivationTransport,
    normalize_activation_transport,
)
from src.quantization.activation_transport_runtime import (  # noqa: E402
    ActivationTransportRuntime,
)
from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer  # noqa: E402


@dataclass(frozen=True)
class StageFormatSelection:
    """One producer transmission's exact candidate table and per-chunk IDs."""

    candidate_formats: tuple[str, ...]
    format_ids: bytes


BatchPlan = dict[str, list[StageFormatSelection]]


class BatchRecordingDynamicInputQuantizer(DynamicInputQuantizer):
    """Dynamic quantizer that records exact per-batch chunk-format choices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_batch_plan: BatchPlan = {}

    def begin_batch(self):
        self.current_batch_plan = {}

    def consume_batch_plan(self) -> BatchPlan:
        plan = self.current_batch_plan
        self.current_batch_plan = {}
        return plan

    def _quantize_input_tensor(self, tensor, layer_name, candidates, module=None):
        quantized, best_indices = super()._quantize_input_tensor(
            tensor,
            layer_name,
            candidates,
            module,
        )
        candidate_formats = tuple(str(fmt) for fmt in candidates)
        if len(candidate_formats) > 256:
            raise ValueError(
                "Replay capture supports at most 256 candidate formats per producer; "
                f"got {len(candidate_formats)} for {layer_name!r}."
            )

        indices = best_indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if indices.numel():
            minimum = int(indices.min().item())
            maximum = int(indices.max().item())
            if minimum < 0 or maximum >= len(candidate_formats):
                raise RuntimeError(
                    f"Selector returned format IDs [{minimum}, {maximum}] for "
                    f"{len(candidate_formats)} candidates at producer {layer_name!r}."
                )
        selection = StageFormatSelection(
            candidate_formats=candidate_formats,
            format_ids=bytes(indices.tolist()),
        )
        self.current_batch_plan.setdefault(layer_name, []).append(selection)
        return quantized, best_indices


class BatchReplayInputQuantizer:
    """Replay exact per-batch producer-stage format choices through transport."""

    def __init__(
        self,
        model,
        chunk_size=128,
        candidate_formats=None,
        transport="encoded",
    ):
        self.model = model
        self.chunk_size = int(chunk_size)
        self.candidate_formats = list(candidate_formats or DEFAULT_CANDIDATE_FORMATS)
        self.transport = normalize_activation_transport(transport)
        self.activation_transport = None
        self._transport_runtime = None
        self.current_batch_plan: BatchPlan = {}
        self.current_offsets: dict[str, int] = defaultdict(int)
        self._pending_stats = defaultdict(deque)
        self.layer_stats = {}
        self.stats = {
            "sum_mse_err": None,
            "sum_l2_norm": None,
        }

    def register_hooks(self):
        """Install producer-stage replay; retained name keeps the utility API stable."""
        if self._transport_runtime is not None:
            return

        self.activation_transport = ActivationTransport(
            mode=self.transport,
            chunk_size=self.chunk_size,
        )

        def encode_stage(stage, tensor):
            selection = self._consume_stage_selection(stage.stage_id)
            format_ids = torch.tensor(
                list(selection.format_ids),
                dtype=torch.int32,
                device=tensor.device,
            )
            transmitted = self.activation_transport.transmit_dynamic(
                tensor,
                format_ids,
                selection.candidate_formats,
                producer_id=stage.stage_id,
            )
            self._pending_stats[stage.stage_id].append(
                (stage, tensor.detach(), selection)
            )
            return transmitted

        def observe_decode(stage_id, quantized):
            pending = self._pending_stats[stage_id]
            if not pending:
                return
            stage, tensor, selection = pending.popleft()
            self._update_stats(stage, tensor, quantized, selection)

        self._transport_runtime = ActivationTransportRuntime(
            self.model,
            self.activation_transport,
            encode_stage,
            decode_observer=observe_decode,
        ).install()

    def load_batch_plan(self, batch_plan: BatchPlan):
        self.current_batch_plan = batch_plan
        self.current_offsets = defaultdict(int)

    def assert_batch_fully_consumed(self):
        leftovers = []
        for stage_id, selections in self.current_batch_plan.items():
            used = self.current_offsets.get(stage_id, 0)
            if used != len(selections):
                leftovers.append(f"{stage_id}: used {used}/{len(selections)}")
        if leftovers:
            raise RuntimeError(
                "Replay did not consume all recorded producer transmissions: "
                + "; ".join(leftovers[:10])
            )

    def _consume_stage_selection(self, stage_id: str) -> StageFormatSelection:
        stage_plan = self.current_batch_plan.get(stage_id)
        if stage_plan is None:
            raise RuntimeError(
                f"No replay plan found for producer stage {stage_id!r} in current batch."
            )
        offset = self.current_offsets[stage_id]
        if offset >= len(stage_plan):
            raise RuntimeError(
                f"Replay plan exhausted early for producer stage {stage_id!r}."
            )
        self.current_offsets[stage_id] += 1
        return stage_plan[offset]

    def _update_stats(self, stage, tensor, quantized, selection):
        with torch.no_grad():
            diff = tensor - quantized
            updates = {
                "sum_mse_err": diff.pow(2).sum(),
                "sum_l2_norm": tensor.pow(2).sum(),
            }
            for key, value in updates.items():
                if self.stats[key] is None:
                    self.stats[key] = value.detach()
                else:
                    self.stats[key] += value.detach()

        stage_stats = self.layer_stats.setdefault(
            stage.stage_id,
            {
                "format_counts": {},
                "type": stage.kind.value,
                "producer_nodes": list(stage.node_names),
                "consumer_nodes": list(stage.consumer_nodes),
                "is_unsigned": bool(stage.is_unsigned),
            },
        )
        format_counts = stage_stats["format_counts"]
        for format_id, count in Counter(selection.format_ids).items():
            fmt = selection.candidate_formats[format_id]
            format_counts[fmt] = format_counts.get(fmt, 0) + count

    def get_final_stats(self):
        sum_mse_err = self.stats["sum_mse_err"]
        sum_l2_norm = self.stats["sum_l2_norm"]
        total_mse = sum_mse_err.item() if isinstance(sum_mse_err, torch.Tensor) else 0.0
        total_l2 = sum_l2_norm.item() if isinstance(sum_l2_norm, torch.Tensor) else 0.0
        result = {
            "transport": self.transport,
            "norm_mse": total_mse / total_l2 if total_l2 > 0 else 0.0,
            "total_mse": total_mse,
        }
        if self._transport_runtime is not None:
            result.update(self._transport_runtime.transport_stats())
        return result

    def cleanup(self):
        if self._transport_runtime is not None:
            self._transport_runtime.cleanup()
            self._transport_runtime = None
        self.activation_transport = None
        self._pending_stats.clear()


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
    parser.add_argument("--metric", type=str, default="mse", help="Dynamic metric. Only mse is supported.")
    parser.add_argument(
        "--transport",
        choices=("encoded", "reference"),
        default="encoded",
        help="Activation transport implementation (default: encoded hardware packets)",
    )
    parser.add_argument(
        "--excluded_ops",
        type=str,
        default="",
        help="Comma-separated op names to exclude from quantization (default: none)",
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
            transport=args.transport,
        )
        replay_quantizer = BatchReplayInputQuantizer(
            model=replay_model,
            chunk_size=args.chunk_size,
            candidate_formats=args.candidate_formats,
            transport=args.transport,
        )
        dynamic_quantizer.register_hooks()
        replay_quantizer.register_hooks()

        dynamic_stage_ids = tuple(
            stage.stage_id for stage in dynamic_quantizer._transport_runtime.plan.stages
        )
        replay_stage_ids = tuple(
            stage.stage_id for stage in replay_quantizer._transport_runtime.plan.stages
        )
        if dynamic_stage_ids != replay_stage_ids:
            raise RuntimeError(
                "Capture and replay models produced different activation-stage plans: "
                f"capture={dynamic_stage_ids!r}, replay={replay_stage_ids!r}"
            )

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

                for stage_selections in batch_plan.values():
                    for selection in stage_selections:
                        for format_id, count in Counter(selection.format_ids).items():
                            aggregate_counts[selection.candidate_formats[format_id]] += count

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
        print(f"Transport: {args.transport}")
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
            f"Dynamic norm_mse={dynamic_stats['norm_mse']:.6e}"
        )
        print(
            f"Replay  norm_mse={replay_stats['norm_mse']:.6e}"
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
