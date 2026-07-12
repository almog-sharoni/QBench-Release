import argparse
import csv
from dataclasses import dataclass
import gc
import json
import math
import os
import struct
import sys
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.pseudo_mse3.pseudo_mse import (  # noqa: E402
    BASELINE_METRIC_NAME,
    MetricComparisonSpec,
    build_pseudo_mse_input_quant_cfg,
    build_pseudo_mse_runtime_config,
    candidate_formats_for_bit_width,
)
from runspace.experiments.utils.common import run_inference  # noqa: E402
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    _parse_fp_format,
    normalize_pseudo_mse3_fixed_rounding,
    normalize_pseudo_mse3_tie_break,
    pseudo_mse_encode_emb_python,
    pseudo_mse3_fixed_point_from_diff,
    pseudo_mse3_fixed_rounding_code,
    pseudo_mse3_tie_break_code,
    pseudo_mse_reconstruct_scaled_python,
    validate_pseudo_mse_candidate_pairs,
)
from runspace.src.quantization.dynamic_input_quantizer import (  # noqa: E402
    DynamicInputQuantizer,
)


DEFAULT_BIT_WIDTHS = (8, 7, 6, 5, 4)
DEFAULT_BITS_TO_TAKE = (0, 1, 3, 5, 7, 9)
SUMMARY_FILENAME = "summary.csv"
MISMATCH_SUMMARY_FILENAME = "mismatch_summary.csv"
FORMAT_CHOICES_PLOT_FILENAME = "format_choices_mse_vs_pseudo_mse3.png"
VECTORS_CSV_FILENAME = "vectors.csv"
VECTORS_TXT_FILENAME = "vectors.txt"
MANIFEST_FILENAME = "manifest.json"
_PRIORITY_MODULUS = 2_147_483_647


def _parse_int_csv(value, *, name, minimum=0):
    if isinstance(value, (list, tuple)):
        items = [int(item) for item in value]
    else:
        items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item < minimum for item in items):
        raise ValueError(f"{name} values must be >= {minimum}; got {items}")
    return list(dict.fromkeys(items))


def _float32_hex(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f"{bits:08x}"


def _hex_to_float32(value):
    return struct.unpack("<f", struct.pack("<I", int(value, 16)))[0]


def _choice_label(choose_exp2):
    return "exp2" if bool(choose_exp2) else "exp1"


def _choice_index(choose_exp2, pair):
    return pair.e2_index if bool(choose_exp2) else pair.e1_index


@dataclass
class BaseChunkAnalysis:
    pair: object
    scales: torch.Tensor
    scaled_chunks: torch.Tensor
    q1_scaled: torch.Tensor
    q2_scaled: torch.Tensor
    err1_pre_square: torch.Tensor
    err2_pre_square: torch.Tensor
    err1_sq: torch.Tensor
    err2_sq: torch.Tensor
    err1_sum: torch.Tensor
    err2_sum: torch.Tensor
    exact_diff: torch.Tensor
    exact_sum: torch.Tensor
    reference_choose_exp2: torch.Tensor
    exact_choose_exp2: torch.Tensor


@dataclass
class FixedChunkAnalysis:
    contributions: torch.Tensor
    chunk_sum: torch.Tensor
    choose_exp2: torch.Tensor


def analyze_chunks(ref_chunks, candidates):
    """Compute the two-candidate MSE reference on already formed raw chunks."""
    pair = validate_pseudo_mse_candidate_pairs(candidates)[0]
    chunks = ref_chunks.to(torch.float32).contiguous()
    scales = DynamicInputQuantizer._chunk_scale(chunks)
    scaled_chunks = chunks / scales
    q1_scaled = pseudo_mse_reconstruct_scaled_python(
        scaled_chunks,
        exp_bits=1,
        mantissa_bits=pair.exp1_mantissa_width,
        is_signed=pair.is_signed,
    )
    q2_scaled = pseudo_mse_reconstruct_scaled_python(
        scaled_chunks,
        exp_bits=2,
        mantissa_bits=pair.exp1_mantissa_width - 1,
        is_signed=pair.is_signed,
    )
    err1_pre_square = scaled_chunks - q1_scaled
    err2_pre_square = scaled_chunks - q2_scaled
    err1_sq = err1_pre_square.pow(2)
    err2_sq = err2_pre_square.pow(2)
    err1_sum = err1_sq.sum(dim=1)
    err2_sum = err2_sq.sum(dim=1)
    exact_diff = err2_sq - err1_sq
    exact_sum = exact_diff.sum(dim=1)
    return BaseChunkAnalysis(
        pair=pair,
        scales=scales,
        scaled_chunks=scaled_chunks,
        q1_scaled=q1_scaled,
        q2_scaled=q2_scaled,
        err1_pre_square=err1_pre_square,
        err2_pre_square=err2_pre_square,
        err1_sq=err1_sq,
        err2_sq=err2_sq,
        err1_sum=err1_sum,
        err2_sum=err2_sum,
        exact_diff=exact_diff,
        exact_sum=exact_sum,
        reference_choose_exp2=err2_sum < err1_sum,
        exact_choose_exp2=exact_sum < 0,
    )


def fixed_analysis_from_diff(
    exact_diff,
    bits_to_take,
    fixed_rounding="floor",
    tie_break="exp1",
):
    """Apply pseudo_MSE3's exact or per-element fixed-point accumulation."""
    contributions = pseudo_mse3_fixed_point_from_diff(
        exact_diff,
        bits_to_take,
        fixed_rounding=fixed_rounding,
    )
    if int(bits_to_take) == 0:
        chunk_sum = contributions.sum(dim=1)
    else:
        chunk_sum = contributions.sum(dim=1, dtype=torch.int64)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    choose_exp2 = chunk_sum <= 0 if tie_break == "exp2" else chunk_sum < 0
    return FixedChunkAnalysis(
        contributions=contributions,
        chunk_sum=chunk_sum,
        choose_exp2=choose_exp2,
    )


@dataclass(frozen=True)
class StatsKey:
    scope: str
    bit_width: int
    bits_to_take: int
    fixed_rounding: str
    tie_break: str
    layer_name: str
    candidate_e1: str
    candidate_e2: str
    signedness: str


@dataclass
class StatsBucket:
    total_chunks: int = 0
    runtime_mse_e1: int = 0
    runtime_mse_e2: int = 0
    reference_mse_e1: int = 0
    reference_mse_e2: int = 0
    pseudo_e1: int = 0
    pseudo_e2: int = 0
    runtime_vs_reference_mismatches: int = 0
    exact_pairwise_vs_reference_mismatches: int = 0
    decision_mismatches: int = 0
    e1_to_e2_mismatches: int = 0
    e2_to_e1_mismatches: int = 0
    mse_ties: int = 0
    exact_pairwise_ties: int = 0
    fixed_ties: int = 0
    sum_abs_exact_margin: float = 0.0
    min_abs_exact_margin: float = math.inf
    max_abs_exact_margin: float = 0.0
    sum_excess_mse: float = 0.0
    max_excess_mse: float = 0.0

    def update(self, base, fixed, runtime_choose_exp2):
        count = int(runtime_choose_exp2.numel())
        reference = base.reference_choose_exp2
        exact = base.exact_choose_exp2
        pseudo = fixed.choose_exp2
        mismatch = pseudo != runtime_choose_exp2
        margin = (base.err2_sum - base.err1_sum).abs()
        selected_error = torch.where(pseudo, base.err2_sum, base.err1_sum)
        excess_mse = selected_error - torch.minimum(base.err1_sum, base.err2_sum)

        self.total_chunks += count
        self.runtime_mse_e2 += int(runtime_choose_exp2.sum().item())
        self.runtime_mse_e1 += count - int(runtime_choose_exp2.sum().item())
        self.reference_mse_e2 += int(reference.sum().item())
        self.reference_mse_e1 += count - int(reference.sum().item())
        self.pseudo_e2 += int(pseudo.sum().item())
        self.pseudo_e1 += count - int(pseudo.sum().item())
        self.runtime_vs_reference_mismatches += int(
            (runtime_choose_exp2 != reference).sum().item()
        )
        self.exact_pairwise_vs_reference_mismatches += int((exact != reference).sum().item())
        self.decision_mismatches += int(mismatch.sum().item())
        self.e1_to_e2_mismatches += int(
            ((~runtime_choose_exp2) & pseudo).sum().item()
        )
        self.e2_to_e1_mismatches += int(
            (runtime_choose_exp2 & (~pseudo)).sum().item()
        )
        self.mse_ties += int((base.err1_sum == base.err2_sum).sum().item())
        self.exact_pairwise_ties += int((base.exact_sum == 0).sum().item())
        self.fixed_ties += int((fixed.chunk_sum == 0).sum().item())
        self.sum_abs_exact_margin += float(margin.sum().item())
        if count:
            self.min_abs_exact_margin = min(
                self.min_abs_exact_margin,
                float(margin.min().item()),
            )
            self.max_abs_exact_margin = max(
                self.max_abs_exact_margin,
                float(margin.max().item()),
            )
            self.sum_excess_mse += float(excess_mse.sum().item())
            self.max_excess_mse = max(
                self.max_excess_mse,
                float(excess_mse.max().item()),
            )

    def as_row(self, key):
        total = self.total_chunks
        return {
            "scope": key.scope,
            "layer_name": key.layer_name,
            "candidate_e1": key.candidate_e1,
            "candidate_e2": key.candidate_e2,
            "signedness": key.signedness,
            "bit_width": key.bit_width,
            "bits_to_take": key.bits_to_take,
            "fixed_rounding": key.fixed_rounding,
            "tie_break": key.tie_break,
            "total_chunks": total,
            "runtime_mse_e1": self.runtime_mse_e1,
            "runtime_mse_e2": self.runtime_mse_e2,
            "reference_mse_e1": self.reference_mse_e1,
            "reference_mse_e2": self.reference_mse_e2,
            "pseudo_e1": self.pseudo_e1,
            "pseudo_e2": self.pseudo_e2,
            "runtime_vs_reference_mismatches": self.runtime_vs_reference_mismatches,
            "exact_pairwise_vs_reference_mismatches": (
                self.exact_pairwise_vs_reference_mismatches
            ),
            "decision_mismatches": self.decision_mismatches,
            "mismatch_rate": self.decision_mismatches / total if total else 0.0,
            "e1_to_e2_mismatches": self.e1_to_e2_mismatches,
            "e2_to_e1_mismatches": self.e2_to_e1_mismatches,
            "mse_ties": self.mse_ties,
            "exact_pairwise_ties": self.exact_pairwise_ties,
            "fixed_ties": self.fixed_ties,
            "mean_abs_exact_margin": self.sum_abs_exact_margin / total if total else 0.0,
            "min_abs_exact_margin": (
                self.min_abs_exact_margin if total else 0.0
            ),
            "max_abs_exact_margin": self.max_abs_exact_margin,
            "mean_excess_mse": self.sum_excess_mse / total if total else 0.0,
            "max_excess_mse": self.max_excess_mse,
        }


@dataclass
class VectorSample:
    priority: int
    category: str
    bit_width: int
    bits_to_take: int
    fixed_rounding: str
    tie_break: str
    layer_name: str
    layer_call_index: int
    local_chunk_index: int
    global_chunk_index: int
    candidates: tuple
    raw_chunk: torch.Tensor
    runtime_mse_choice: str
    capture_reference_mse_choice: str
    capture_exact_pairwise_choice: str
    capture_pseudo_choice: str


class PriorityReservoir:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.items = []

    def consider(self, sample):
        if self.capacity <= 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(sample)
            return
        worst_index = max(range(len(self.items)), key=lambda idx: self.items[idx].priority)
        if sample.priority < self.items[worst_index].priority:
            self.items[worst_index] = sample

    def sorted_items(self):
        return sorted(self.items, key=lambda sample: sample.priority)


class RealActivationCollector:
    def __init__(
        self,
        bits_to_take=DEFAULT_BITS_TO_TAKE,
        fixed_rounding="floor",
        tie_break="exp1",
        analysis_chunks_per_batch=8192,
        max_mismatch_vectors=10,
        control_vectors=2,
        seed=42,
    ):
        self.bits_to_take = tuple(int(value) for value in bits_to_take)
        self.fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
        self.tie_break = normalize_pseudo_mse3_tie_break(tie_break)
        self.analysis_chunks_per_batch = int(analysis_chunks_per_batch)
        self.max_mismatch_vectors = int(max_mismatch_vectors)
        self.control_vectors = int(control_vectors)
        self.seed = int(seed)
        if self.analysis_chunks_per_batch < 1:
            raise ValueError("analysis_chunks_per_batch must be at least 1")
        if any(value < 0 for value in self.bits_to_take):
            raise ValueError("bits_to_take values must be non-negative")
        if self.max_mismatch_vectors < 0 or self.control_vectors < 0:
            raise ValueError("vector limits must be non-negative")

        self._stats = {}
        self._reservoirs = {}
        self._next_chunk_ordinal = {}
        self._layer_call_counts = {}
        self.observer_calls = 0
        self.candidate_pairs = set()

    def _stats_bucket(self, key):
        if key not in self._stats:
            self._stats[key] = StatsBucket()
        return self._stats[key]

    def _reservoir(self, bit_width, bits_to_take, category):
        key = (
            int(bit_width),
            int(bits_to_take),
            self.fixed_rounding,
            self.tie_break,
            str(category),
        )
        if key not in self._reservoirs:
            capacity = (
                self.max_mismatch_vectors
                if category == "mismatch"
                else self.control_vectors
            )
            self._reservoirs[key] = PriorityReservoir(capacity)
        return self._reservoirs[key]

    def _priorities(self, ordinals, bits_to_take, category):
        salt = (
            self.seed * 1009
            + int(bits_to_take) * 9176
            + (31337 if category == "mismatch" else 7331)
        )
        values = torch.remainder(ordinals + salt, _PRIORITY_MODULUS)
        values = torch.remainder(values * 48271 + 1, _PRIORITY_MODULUS)
        values = torch.remainder(values * 69621 + 1, _PRIORITY_MODULUS)
        return values

    def _sample_category(
        self,
        *,
        category,
        mask,
        base,
        fixed,
        runtime_choose_exp2,
        raw_chunks,
        candidates,
        layer_name,
        layer_call_index,
        local_offset,
        global_offset,
        bits_to_take,
    ):
        reservoir = self._reservoir(base.pair.bit_width, bits_to_take, category)
        if reservoir.capacity <= 0:
            return
        eligible_count = int(mask.sum().item())
        if eligible_count == 0:
            return

        chunk_count = int(mask.numel())
        ordinals = torch.arange(
            global_offset,
            global_offset + chunk_count,
            dtype=torch.int64,
            device=mask.device,
        )
        priorities = self._priorities(ordinals, bits_to_take, category)
        masked_priorities = torch.where(
            mask,
            priorities,
            torch.full_like(priorities, _PRIORITY_MODULUS),
        )
        take = min(reservoir.capacity, eligible_count)
        selected_priorities, selected_indices = torch.topk(
            masked_priorities,
            k=take,
            largest=False,
            sorted=False,
        )

        indices = selected_indices.detach().cpu().tolist()
        priority_values = selected_priorities.detach().cpu().tolist()
        raw_selected = raw_chunks[selected_indices].detach().cpu()
        runtime_selected = runtime_choose_exp2[selected_indices].detach().cpu().tolist()
        reference_selected = base.reference_choose_exp2[selected_indices].detach().cpu().tolist()
        exact_selected = base.exact_choose_exp2[selected_indices].detach().cpu().tolist()
        pseudo_selected = fixed.choose_exp2[selected_indices].detach().cpu().tolist()

        for position, local_index in enumerate(indices):
            sample = VectorSample(
                priority=int(priority_values[position]),
                category=category,
                bit_width=int(base.pair.bit_width),
                bits_to_take=int(bits_to_take),
                fixed_rounding=self.fixed_rounding,
                tie_break=self.tie_break,
                layer_name=str(layer_name),
                layer_call_index=int(layer_call_index),
                local_chunk_index=int(local_offset + local_index),
                global_chunk_index=int(global_offset + local_index),
                candidates=tuple(candidates),
                raw_chunk=raw_selected[position].contiguous().clone(),
                runtime_mse_choice=_choice_label(runtime_selected[position]),
                capture_reference_mse_choice=_choice_label(reference_selected[position]),
                capture_exact_pairwise_choice=_choice_label(exact_selected[position]),
                capture_pseudo_choice=_choice_label(pseudo_selected[position]),
            )
            reservoir.consider(sample)

    def observe(self, *, layer_name, candidates, ref_chunks, best_indices):
        """Consume one synchronous DynamicInputQuantizer chunk observation."""
        pair = validate_pseudo_mse_candidate_pairs(candidates)[0]
        if ref_chunks.ndim != 2:
            raise ValueError(
                f"Expected 2D chunks for {layer_name!r}; got shape {tuple(ref_chunks.shape)}"
            )
        if int(ref_chunks.shape[0]) != int(best_indices.numel()):
            raise ValueError(
                f"Chunk/index count mismatch for {layer_name!r}: "
                f"{ref_chunks.shape[0]} chunks vs {best_indices.numel()} indices"
            )

        bit_width = int(pair.bit_width)
        pair_names = (pair.e1_format, pair.e2_format)
        self.candidate_pairs.add(pair_names)
        call_key = (bit_width, str(layer_name))
        layer_call_index = self._layer_call_counts.get(call_key, 0)
        self._layer_call_counts[call_key] = layer_call_index + 1
        global_base = self._next_chunk_ordinal.get(bit_width, 0)
        total_chunks = int(ref_chunks.shape[0])
        self._next_chunk_ordinal[bit_width] = global_base + total_chunks
        self.observer_calls += 1

        signedness = "signed" if pair.is_signed else "unsigned"
        for start in range(0, total_chunks, self.analysis_chunks_per_batch):
            end = min(start + self.analysis_chunks_per_batch, total_chunks)
            raw_batch = ref_chunks[start:end]
            index_batch = best_indices[start:end]
            base = analyze_chunks(raw_batch, candidates)
            runtime_choose_exp2 = index_batch == pair.e2_index

            for bits_to_take in self.bits_to_take:
                fixed = fixed_analysis_from_diff(
                    base.exact_diff,
                    bits_to_take,
                    fixed_rounding=self.fixed_rounding,
                    tie_break=self.tie_break,
                )
                global_key = StatsKey(
                    scope="global",
                    bit_width=bit_width,
                    bits_to_take=bits_to_take,
                    fixed_rounding=self.fixed_rounding,
                    tie_break=self.tie_break,
                    layer_name="__all__",
                    candidate_e1="mixed",
                    candidate_e2="mixed",
                    signedness="mixed",
                )
                layer_key = StatsKey(
                    scope="layer",
                    bit_width=bit_width,
                    bits_to_take=bits_to_take,
                    fixed_rounding=self.fixed_rounding,
                    tie_break=self.tie_break,
                    layer_name=str(layer_name),
                    candidate_e1=pair.e1_format,
                    candidate_e2=pair.e2_format,
                    signedness=signedness,
                )
                self._stats_bucket(global_key).update(base, fixed, runtime_choose_exp2)
                self._stats_bucket(layer_key).update(base, fixed, runtime_choose_exp2)

                mismatch = fixed.choose_exp2 != runtime_choose_exp2
                common = {
                    "base": base,
                    "fixed": fixed,
                    "runtime_choose_exp2": runtime_choose_exp2,
                    "raw_chunks": raw_batch,
                    "candidates": candidates,
                    "layer_name": layer_name,
                    "layer_call_index": layer_call_index,
                    "local_offset": start,
                    "global_offset": global_base + start,
                    "bits_to_take": bits_to_take,
                }
                self._sample_category(
                    category="mismatch",
                    mask=mismatch,
                    **common,
                )
                self._sample_category(
                    category="control",
                    mask=~mismatch,
                    **common,
                )

    def summary_rows(self):
        def sort_key(item):
            key, _bucket = item
            scope_order = 0 if key.scope == "global" else 1
            return (
                -key.bit_width,
                key.bits_to_take,
                key.fixed_rounding,
                key.tie_break,
                scope_order,
                key.layer_name,
            )

        return [bucket.as_row(key) for key, bucket in sorted(self._stats.items(), key=sort_key)]

    def samples(self):
        category_order = {"mismatch": 0, "control": 1}
        samples = []
        for key in sorted(
            self._reservoirs,
            key=lambda value: (
                -value[0],
                value[1],
                value[2],
                value[3],
                category_order[value[4]],
            ),
        ):
            samples.extend(self._reservoirs[key].sorted_items())
        return samples


VECTOR_FIELDS = [
    "sample_id",
    "category",
    "model",
    "layer_name",
    "layer_call_index",
    "local_chunk_index",
    "global_chunk_index",
    "bit_width",
    "bits_to_take",
    "fixed_rounding",
    "tie_break",
    "candidate_e1",
    "candidate_e2",
    "signedness",
    "runtime_mse_choice",
    "capture_reference_mse_choice",
    "capture_exact_pairwise_choice",
    "capture_pseudo_choice",
    "reference_mse_choice",
    "exact_pairwise_choice",
    "pseudo_choice",
    "err1_sum",
    "err2_sum",
    "exact_diff_sum",
    "fixed_sum",
    "chunk_scale_fp32_hex",
    "chunk_scale",
    "value_index",
    "raw_fp32_hex",
    "raw_value",
    "scaled_fp32_hex",
    "scaled_value",
    "q_e1_bits",
    "q_e1_value",
    "q_e2_bits",
    "q_e2_value",
    "err_e1_fp32_hex",
    "err_e1",
    "err_e2_fp32_hex",
    "err_e2",
    "err1_sq_fp32_hex",
    "err1_sq",
    "err2_sq_fp32_hex",
    "err2_sq",
    "exact_diff_fp32_hex",
    "exact_diff",
    "fixed_contribution",
]


def _vector_rows(sample, sample_id, model_name):
    raw = sample.raw_chunk.to(torch.float32).reshape(1, -1)
    base = analyze_chunks(raw, sample.candidates)
    fixed = fixed_analysis_from_diff(
        base.exact_diff,
        sample.bits_to_take,
        fixed_rounding=sample.fixed_rounding,
        tie_break=sample.tie_break,
    )
    pair = base.pair
    q1_bits = pseudo_mse_encode_emb_python(
        base.scaled_chunks,
        exp_bits=1,
        mantissa_bits=pair.exp1_mantissa_width,
        is_signed=pair.is_signed,
    )
    q2_bits = pseudo_mse_encode_emb_python(
        base.scaled_chunks,
        exp_bits=2,
        mantissa_bits=pair.exp1_mantissa_width - 1,
        is_signed=pair.is_signed,
    )
    reference_choice = _choice_label(base.reference_choose_exp2[0])
    exact_choice = _choice_label(base.exact_choose_exp2[0])
    pseudo_choice = _choice_label(fixed.choose_exp2[0])
    fixed_sum = fixed.chunk_sum[0].item()
    fixed_contributions = fixed.contributions[0]

    rows = []
    for value_index in range(raw.shape[1]):
        fixed_value = fixed_contributions[value_index].item()
        rows.append(
            {
                "sample_id": sample_id,
                "category": sample.category,
                "model": model_name,
                "layer_name": sample.layer_name,
                "layer_call_index": sample.layer_call_index,
                "local_chunk_index": sample.local_chunk_index,
                "global_chunk_index": sample.global_chunk_index,
                "bit_width": sample.bit_width,
                "bits_to_take": sample.bits_to_take,
                "fixed_rounding": sample.fixed_rounding,
                "tie_break": sample.tie_break,
                "candidate_e1": pair.e1_format,
                "candidate_e2": pair.e2_format,
                "signedness": "signed" if pair.is_signed else "unsigned",
                "runtime_mse_choice": sample.runtime_mse_choice,
                "capture_reference_mse_choice": sample.capture_reference_mse_choice,
                "capture_exact_pairwise_choice": sample.capture_exact_pairwise_choice,
                "capture_pseudo_choice": sample.capture_pseudo_choice,
                "reference_mse_choice": reference_choice,
                "exact_pairwise_choice": exact_choice,
                "pseudo_choice": pseudo_choice,
                "err1_sum": float(base.err1_sum[0].item()),
                "err2_sum": float(base.err2_sum[0].item()),
                "exact_diff_sum": float(base.exact_sum[0].item()),
                "fixed_sum": fixed_sum,
                "chunk_scale_fp32_hex": _float32_hex(base.scales[0, 0].item()),
                "chunk_scale": float(base.scales[0, 0].item()),
                "value_index": value_index,
                "raw_fp32_hex": _float32_hex(raw[0, value_index].item()),
                "raw_value": float(raw[0, value_index].item()),
                "scaled_fp32_hex": _float32_hex(base.scaled_chunks[0, value_index].item()),
                "scaled_value": float(base.scaled_chunks[0, value_index].item()),
                "q_e1_bits": format(int(q1_bits[0, value_index].item()), f"0{pair.bit_width}b"),
                "q_e1_value": float(base.q1_scaled[0, value_index].item()),
                "q_e2_bits": format(int(q2_bits[0, value_index].item()), f"0{pair.bit_width}b"),
                "q_e2_value": float(base.q2_scaled[0, value_index].item()),
                "err_e1_fp32_hex": _float32_hex(base.err1_pre_square[0, value_index].item()),
                "err_e1": float(base.err1_pre_square[0, value_index].item()),
                "err_e2_fp32_hex": _float32_hex(base.err2_pre_square[0, value_index].item()),
                "err_e2": float(base.err2_pre_square[0, value_index].item()),
                "err1_sq_fp32_hex": _float32_hex(base.err1_sq[0, value_index].item()),
                "err1_sq": float(base.err1_sq[0, value_index].item()),
                "err2_sq_fp32_hex": _float32_hex(base.err2_sq[0, value_index].item()),
                "err2_sq": float(base.err2_sq[0, value_index].item()),
                "exact_diff_fp32_hex": _float32_hex(base.exact_diff[0, value_index].item()),
                "exact_diff": float(base.exact_diff[0, value_index].item()),
                "fixed_contribution": fixed_value,
            }
        )
    return rows


def _write_summary(path, rows):
    if not rows:
        raise ValueError("No real activation statistics were collected")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_format_choices_plot(mismatch_summary_path, *, model_name=None):
    try:
        from runspace.experiments.pseudo_mse3.plot_format_choices import (
            regenerate_plot,
        )
    except Exception as exc:
        print(f"[plot] skipped pseudo_MSE3 format choices ({exc})")
        return None

    output_path = os.path.join(
        os.path.dirname(mismatch_summary_path),
        FORMAT_CHOICES_PLOT_FILENAME,
    )
    title = "MSE vs Pseudo_MSE3 Format Selections"
    if model_name:
        title = f"{model_name}: {title}"
    return regenerate_plot(
        summary_csv=mismatch_summary_path,
        output_path=output_path,
        title=title,
    )


def _mismatch_summary_rows(summary_rows):
    rows = []
    for summary in summary_rows:
        if summary["scope"] != "global":
            continue
        total_chunks = int(summary["total_chunks"])
        mismatches = int(summary["decision_mismatches"])
        matched_chunks = total_chunks - mismatches
        rows.append(
            {
                "bit_width": int(summary["bit_width"]),
                "bits_to_take": int(summary["bits_to_take"]),
                "fixed_rounding": summary.get("fixed_rounding", "floor"),
                "tie_break": summary.get("tie_break", "exp1"),
                "total_chunks": total_chunks,
                "matched_chunks": matched_chunks,
                "decision_mismatches": mismatches,
                "match_rate": matched_chunks / total_chunks if total_chunks else 0.0,
                "mismatch_rate": mismatches / total_chunks if total_chunks else 0.0,
                "mismatch_percent": (
                    100.0 * mismatches / total_chunks if total_chunks else 0.0
                ),
                "e1_to_e2_mismatches": int(summary["e1_to_e2_mismatches"]),
                "e2_to_e1_mismatches": int(summary["e2_to_e1_mismatches"]),
                "runtime_mse_e1": int(summary["runtime_mse_e1"]),
                "runtime_mse_e2": int(summary["runtime_mse_e2"]),
                "pseudo_e1": int(summary["pseudo_e1"]),
                "pseudo_e2": int(summary["pseudo_e2"]),
                "runtime_vs_reference_mismatches": int(
                    summary["runtime_vs_reference_mismatches"]
                ),
                "exact_pairwise_vs_reference_mismatches": int(
                    summary["exact_pairwise_vs_reference_mismatches"]
                ),
                "mse_ties": int(summary["mse_ties"]),
                "fixed_ties": int(summary["fixed_ties"]),
            }
        )
    return rows


def rebuild_mismatch_summary(artifact_dir):
    summary_path = os.path.join(artifact_dir, SUMMARY_FILENAME)
    mismatch_summary_path = os.path.join(artifact_dir, MISMATCH_SUMMARY_FILENAME)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Detailed summary not found: {summary_path}")
    with open(summary_path, newline="") as handle:
        summary_rows = [
            {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]
    mismatch_rows = _mismatch_summary_rows(summary_rows)
    _write_summary(mismatch_summary_path, mismatch_rows)

    manifest_path = os.path.join(artifact_dir, MANIFEST_FILENAME)
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    model_name = None
    if manifest is not None:
        model_name = manifest.get("model_name") or manifest.get("model")
    plot_path = _write_format_choices_plot(
        mismatch_summary_path,
        model_name=model_name,
    )
    if manifest is not None:
        manifest["mismatch_summary_rows"] = len(mismatch_rows)
        manifest.setdefault("artifacts", {})[
            "mismatch_summary_csv"
        ] = MISMATCH_SUMMARY_FILENAME
        if plot_path:
            manifest["artifacts"][
                "format_choices_plot"
            ] = FORMAT_CHOICES_PLOT_FILENAME
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return mismatch_summary_path


def _write_vectors_csv(path, samples, model_name):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VECTOR_FIELDS)
        writer.writeheader()
        for index, sample in enumerate(samples):
            sample_id = (
                f"bw{sample.bit_width}_n{sample.bits_to_take}_r{sample.fixed_rounding}_"
                f"t{sample.tie_break}_"
                f"{sample.category}_{index:04d}"
            )
            writer.writerows(_vector_rows(sample, sample_id, model_name))


def _write_vectors_txt(path, samples, model_name):
    with open(path, "w") as handle:
        handle.write("# pseudo_MSE3 real-activation hardware vectors\n")
        handle.write(f"# model: {model_name}\n")
        handle.write("# activation source: runtime MSE dynamic-quantization trajectory\n")
        handle.write("# decision: exp2 iff accumulator < 0; ties choose exp1\n")
        handle.write(
            "# value columns: index raw_fp32_hex scaled_fp32_hex "
            "q_e1_bits q_e2_bits err1_fp32_hex err2_fp32_hex "
            "exact_diff_fp32_hex fixed_contribution\n\n"
        )
        for index, sample in enumerate(samples):
            sample_id = (
                f"bw{sample.bit_width}_n{sample.bits_to_take}_r{sample.fixed_rounding}_"
                f"t{sample.tie_break}_"
                f"{sample.category}_{index:04d}"
            )
            rows = _vector_rows(sample, sample_id, model_name)
            first = rows[0]
            handle.write(f"BEGIN_VECTOR {sample_id}\n")
            for key in (
                "category",
                "layer_name",
                "layer_call_index",
                "local_chunk_index",
                "global_chunk_index",
                "candidate_e1",
                "candidate_e2",
                "bits_to_take",
                "fixed_rounding",
                "tie_break",
                "chunk_scale_fp32_hex",
                "chunk_scale",
                "runtime_mse_choice",
                "reference_mse_choice",
                "exact_pairwise_choice",
                "pseudo_choice",
                "err1_sum",
                "err2_sum",
                "exact_diff_sum",
                "fixed_sum",
            ):
                handle.write(f"{key} {first[key]}\n")
            for row in rows:
                handle.write(
                    "value "
                    f"{row['value_index']} {row['raw_fp32_hex']} "
                    f"{row['scaled_fp32_hex']} {row['q_e1_bits']} "
                    f"{row['q_e2_bits']} {row['err1_sq_fp32_hex']} "
                    f"{row['err2_sq_fp32_hex']} {row['exact_diff_fp32_hex']} "
                    f"{row['fixed_contribution']}\n"
                )
            handle.write("END_VECTOR\n\n")


def _validate_vector_group(rows):
    first = rows[0]
    expected_indices = list(range(len(rows)))
    actual_indices = [int(row["value_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise AssertionError(
            f"Vector {first['sample_id']} has non-contiguous value indices"
        )
    raw = torch.tensor(
        [_hex_to_float32(row["raw_fp32_hex"]) for row in rows],
        dtype=torch.float32,
    ).unsqueeze(0)
    candidates = (first["candidate_e1"], first["candidate_e2"])
    bits_to_take = int(first["bits_to_take"])
    fixed_rounding = first.get("fixed_rounding", "floor")
    tie_break = first.get("tie_break", "exp1")
    base = analyze_chunks(raw, candidates)
    fixed = fixed_analysis_from_diff(
        base.exact_diff,
        bits_to_take,
        fixed_rounding=fixed_rounding,
        tie_break=tie_break,
    )
    pair = base.pair
    q1_bits = pseudo_mse_encode_emb_python(
        base.scaled_chunks,
        1,
        pair.exp1_mantissa_width,
        pair.is_signed,
    )
    q2_bits = pseudo_mse_encode_emb_python(
        base.scaled_chunks,
        2,
        pair.exp1_mantissa_width - 1,
        pair.is_signed,
    )
    for index, row in enumerate(rows):
        checks = {
            "scaled_fp32_hex": _float32_hex(base.scaled_chunks[0, index].item()),
            "err1_sq_fp32_hex": _float32_hex(base.err1_sq[0, index].item()),
            "err2_sq_fp32_hex": _float32_hex(base.err2_sq[0, index].item()),
            "exact_diff_fp32_hex": _float32_hex(base.exact_diff[0, index].item()),
            "q_e1_bits": format(int(q1_bits[0, index].item()), f"0{pair.bit_width}b"),
            "q_e2_bits": format(int(q2_bits[0, index].item()), f"0{pair.bit_width}b"),
        }
        for field, expected in checks.items():
            if row[field] != expected:
                raise AssertionError(
                    f"Vector {first['sample_id']} row {index} field {field}: "
                    f"expected {expected}, got {row[field]}"
                )
        expected_fixed = str(fixed.contributions[0, index].item())
        if row["fixed_contribution"] != expected_fixed:
            raise AssertionError(
                f"Vector {first['sample_id']} row {index} fixed contribution: "
                f"expected {expected_fixed}, got {row['fixed_contribution']}"
            )

    decisions = {
        "reference_mse_choice": _choice_label(base.reference_choose_exp2[0]),
        "exact_pairwise_choice": _choice_label(base.exact_choose_exp2[0]),
        "pseudo_choice": _choice_label(fixed.choose_exp2[0]),
    }
    for field, expected in decisions.items():
        if first[field] != expected:
            raise AssertionError(
                f"Vector {first['sample_id']} field {field}: "
                f"expected {expected}, got {first[field]}"
            )


def validate_vectors_csv(path):
    vector_count = 0
    current_id = None
    current_rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if current_id is not None and sample_id != current_id:
                _validate_vector_group(current_rows)
                vector_count += 1
                current_rows = []
            current_id = sample_id
            current_rows.append(row)
    if current_rows:
        _validate_vector_group(current_rows)
        vector_count += 1
    return vector_count


def write_outputs(output_dir, collector, *, model_name, manifest):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, SUMMARY_FILENAME)
    mismatch_summary_path = os.path.join(output_dir, MISMATCH_SUMMARY_FILENAME)
    vectors_csv_path = os.path.join(output_dir, VECTORS_CSV_FILENAME)
    vectors_txt_path = os.path.join(output_dir, VECTORS_TXT_FILENAME)
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    summary_rows = collector.summary_rows()
    mismatch_summary_rows = _mismatch_summary_rows(summary_rows)
    samples = collector.samples()

    _write_summary(summary_path, summary_rows)
    _write_summary(mismatch_summary_path, mismatch_summary_rows)
    format_choices_plot_path = _write_format_choices_plot(
        mismatch_summary_path,
        model_name=model_name,
    )
    _write_vectors_csv(vectors_csv_path, samples, model_name)
    _write_vectors_txt(vectors_txt_path, samples, model_name)
    validated_vectors = validate_vectors_csv(vectors_csv_path)
    if validated_vectors != len(samples):
        raise AssertionError(
            f"Validated {validated_vectors} vectors, expected {len(samples)}"
        )

    vector_counts = {}
    for sample in samples:
        key = (
            f"bw{sample.bit_width}_n{sample.bits_to_take}_"
            f"r{sample.fixed_rounding}_t{sample.tie_break}_{sample.category}"
        )
        vector_counts[key] = vector_counts.get(key, 0) + 1
    artifact_paths = {
        "summary_csv": SUMMARY_FILENAME,
        "mismatch_summary_csv": MISMATCH_SUMMARY_FILENAME,
        "vectors_csv": VECTORS_CSV_FILENAME,
        "vectors_txt": VECTORS_TXT_FILENAME,
    }
    if format_choices_plot_path:
        artifact_paths["format_choices_plot"] = FORMAT_CHOICES_PLOT_FILENAME

    manifest = dict(manifest)
    manifest.update(
        {
            "observer_calls": collector.observer_calls,
            "fixed_rounding": collector.fixed_rounding,
            "tie_break": collector.tie_break,
            "candidate_pairs": [list(pair) for pair in sorted(collector.candidate_pairs)],
            "summary_rows": len(summary_rows),
            "mismatch_summary_rows": len(mismatch_summary_rows),
            "validated_vectors": validated_vectors,
            "vector_counts": vector_counts,
            "artifacts": artifact_paths,
        }
    )
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    paths = {
        "summary": summary_path,
        "mismatch_summary": mismatch_summary_path,
        "vectors_csv": vectors_csv_path,
        "vectors_txt": vectors_txt_path,
        "manifest": manifest_path,
    }
    if format_choices_plot_path:
        paths["format_choices_plot"] = format_choices_plot_path
    return paths


def verify_samples_cuda(samples):
    if not torch.cuda.is_available():
        raise RuntimeError("--verify-cuda requires a CUDA device")
    from runspace.src.quantization.cuda import search_best_chunk_format

    groups = {}
    for sample in samples:
        groups.setdefault(
            (
                sample.candidates,
                sample.bits_to_take,
                sample.fixed_rounding,
                sample.tie_break,
            ),
            [],
        ).append(sample)

    verified = 0
    for (candidates, bits_to_take, fixed_rounding, tie_break), group in groups.items():
        pair = validate_pseudo_mse_candidate_pairs(candidates)[0]
        raw = torch.stack([sample.raw_chunk for sample in group]).to("cuda")
        cands_e = []
        cands_m = []
        cands_sgn = []
        for candidate in candidates:
            _width, exp_bits, mantissa_bits, is_signed = _parse_fp_format(candidate)
            cands_e.append(exp_bits)
            cands_m.append(mantissa_bits)
            cands_sgn.append(1 if is_signed else 0)
        best_indices, scales, quantized_flat, unscaled_flat = search_best_chunk_format(
            raw.reshape(-1).contiguous(),
            torch.tensor(cands_e, dtype=torch.int32, device="cuda"),
            torch.tensor(cands_m, dtype=torch.int32, device="cuda"),
            torch.tensor(cands_sgn, dtype=torch.int32, device="cuda"),
            True,
            9,
            float(bits_to_take),
            0,
            pseudo_mse3_fixed_rounding_code(fixed_rounding),
            pseudo_mse3_tie_break_code(tie_break),
        )
        cpu_base = analyze_chunks(raw.cpu(), candidates)
        cpu_fixed = fixed_analysis_from_diff(
            cpu_base.exact_diff,
            bits_to_take,
            fixed_rounding=fixed_rounding,
            tie_break=tie_break,
        )
        expected_indices = torch.where(
            cpu_fixed.choose_exp2,
            torch.full_like(cpu_fixed.choose_exp2, pair.e2_index, dtype=torch.long),
            torch.full_like(cpu_fixed.choose_exp2, pair.e1_index, dtype=torch.long),
        )
        if not torch.equal(best_indices.cpu(), expected_indices):
            bad = torch.nonzero(best_indices.cpu() != expected_indices).flatten().tolist()
            raise AssertionError(
                f"CUDA decision mismatch for {candidates}, bits_to_take={bits_to_take}, "
                f"fixed_rounding={fixed_rounding}, "
                f"tie_break={tie_break}, "
                f"sample indices={bad}"
            )
        expected_unscaled = torch.where(
            cpu_fixed.choose_exp2.unsqueeze(1),
            cpu_base.q2_scaled,
            cpu_base.q1_scaled,
        )
        expected_quantized = expected_unscaled * cpu_base.scales
        torch.testing.assert_close(scales.cpu().view(-1, 1), cpu_base.scales, rtol=0, atol=0)
        torch.testing.assert_close(
            unscaled_flat.cpu().view_as(expected_unscaled),
            expected_unscaled,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            quantized_flat.cpu().view_as(expected_quantized),
            expected_quantized,
            rtol=0,
            atol=0,
        )
        verified += len(group)
    return verified


def _metric_spec(bit_width):
    candidates = candidate_formats_for_bit_width(bit_width, (1, 2))
    return MetricComparisonSpec(
        bit_width=int(bit_width),
        activation_dt=f"dyn_a{int(bit_width)}_e1e2_mse_real_vectors",
        candidate_formats=candidates,
        metric=BASELINE_METRIC_NAME,
        metric_label="MSE",
        metric_slug="mse",
    )


def _artifact_dir(args):
    path = os.path.join(args.output_dir, args.model_name)
    mode_parts = []
    if args.fixed_rounding != "floor":
        mode_parts.append(args.fixed_rounding)
    if args.tie_break != "exp1":
        mode_parts.append(f"tie_{args.tie_break}")
    if mode_parts:
        path = os.path.join(path, "_".join(mode_parts))
    return path


def run_real_data_vectors(args, device=None):
    from runspace.core.runner import Runner

    if device is None:
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
    runner = Runner(device)
    specs = [_metric_spec(bit_width) for bit_width in args.bit_widths]
    collector = RealActivationCollector(
        bits_to_take=args.bits_to_take_values,
        fixed_rounding=args.fixed_rounding,
        tie_break=args.tie_break,
        analysis_chunks_per_batch=args.analysis_chunks_per_batch,
        max_mismatch_vectors=args.max_mismatch_vectors,
        control_vectors=args.control_vectors,
        seed=args.random_seed,
    )
    artifact_dir = _artifact_dir(args)
    model_cache_dir = os.path.join(artifact_dir, "model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)

    def loader_config_builder(current_args):
        return build_pseudo_mse_runtime_config(
            current_args,
            specs[0],
            model_name=current_args.model_name,
            weights=current_args.weights,
        )

    loader = runner.setup_data_loader(loader_config_builder(args))
    trajectory_results = []
    for spec in specs:
        print(
            f"\n[real vectors] bit_width={spec.bit_width} "
            f"candidates={spec.candidate_formats}"
        )
        config = build_pseudo_mse_runtime_config(
            args,
            spec,
            model_name=args.model_name,
            weights=args.weights,
        )
        input_quant_cfg = build_pseudo_mse_input_quant_cfg(
            args,
            spec,
            model_name=args.model_name,
        )
        input_quant_cfg["collect_error_stats"] = False
        input_quant_cfg["collect_format_stats"] = False
        input_quant_cfg["pseudo_mse3_fixed_rounding"] = args.fixed_rounding
        input_quant_cfg["pseudo_mse3_tie_break"] = args.tie_break
        input_quant_cfg["chunk_observer"] = collector.observe
        model = None
        adapter = None
        calls_before = collector.observer_calls
        try:
            model, adapter, _ = runner.prepare_model_with_materialized_weights(
                config=config,
                output_dir=model_cache_dir,
            )
            acc1, acc5, certainty, _stats = run_inference(
                runner,
                model,
                adapter,
                loader,
                args,
                input_quant_cfg=input_quant_cfg,
                desc=f"real-vectors/fp{spec.bit_width}",
            )
            calls = collector.observer_calls - calls_before
            if calls <= 0:
                raise RuntimeError(
                    f"No dynamic-input chunks were observed for bit width {spec.bit_width}"
                )
            trajectory_results.append(
                {
                    "bit_width": spec.bit_width,
                    "candidate_formats": spec.candidate_formats,
                    "observer_calls": calls,
                    "acc1": acc1,
                    "acc5": acc5,
                    "certainty": certainty,
                }
            )
        finally:
            if model is not None:
                del model
            if adapter is not None:
                del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    samples = collector.samples()
    cuda_verified_vectors = 0
    if args.verify_cuda:
        cuda_verified_vectors = verify_samples_cuda(samples)

    manifest = {
        "model": args.model_name,
        "weights": args.weights,
        "model_source": args.model_source,
        "dataset": args.dataset_name,
        "dataset_path": args.dataset_path,
        "activation_source": "mse_dynamic_trajectory",
        "batch_size": args.batch_size,
        "limit_batches": args.limit_batches,
        "random_seed": args.random_seed,
        "chunk_size": args.chunk_size,
        "bit_widths": args.bit_widths,
        "bits_to_take": args.bits_to_take_values,
        "fixed_rounding": args.fixed_rounding,
        "tie_break": args.tie_break,
        "analysis_chunks_per_batch": args.analysis_chunks_per_batch,
        "max_mismatch_vectors": args.max_mismatch_vectors,
        "control_vectors": args.control_vectors,
        "cuda_verified_vectors": cuda_verified_vectors,
        "trajectory_results": trajectory_results,
    }
    paths = write_outputs(
        artifact_dir,
        collector,
        model_name=args.model_name,
        manifest=manifest,
    )
    print("\nReal-activation pseudo_MSE3 artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare pseudo_MSE3 fixed-point decisions with MSE on real model "
            "activation chunks and export bounded hardware vectors."
        )
    )
    parser.add_argument("--model-name", "--model_name", default="resnet50")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--model-source", "--model_source", default="auto")
    parser.add_argument("--dataset-name", "--dataset_name", default="imagenet")
    parser.add_argument("--dataset-path", "--dataset_path", default="/data/imagenet/val")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=16)
    parser.add_argument("--limit-batches", "--limit_batches", type=int, default=1)
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=128)
    parser.add_argument(
        "--bit-widths",
        "--bit_widths",
        default=",".join(str(value) for value in DEFAULT_BIT_WIDTHS),
    )
    parser.add_argument(
        "--bits-to-take",
        "--bits_to_take",
        default=",".join(str(value) for value in DEFAULT_BITS_TO_TAKE),
    )
    parser.add_argument(
        "--fixed-rounding",
        "--fixed_rounding",
        type=normalize_pseudo_mse3_fixed_rounding,
        choices=("floor", "nearest"),
        default="floor",
        help=(
            "Fixed-point conversion. nearest matches activation round-to-nearest "
            "with exact half away from zero."
        ),
    )
    parser.add_argument(
        "--tie-break",
        "--tie_break",
        type=normalize_pseudo_mse3_tie_break,
        choices=("exp1", "exp2"),
        default="exp1",
        help="Chunk-sum tie policy: exp1 uses < 0; exp2 uses <= 0.",
    )
    parser.add_argument("--random-seed", "--random_seed", type=int, default=42)
    parser.add_argument("--random-subset-size", "--random_subset_size", type=int, default=-1)
    parser.add_argument(
        "--analysis-chunks-per-batch",
        type=int,
        default=8192,
        help="Maximum chunks analyzed at once to bound temporary GPU memory.",
    )
    parser.add_argument("--max-mismatch-vectors", type=int, default=10)
    parser.add_argument("--control-vectors", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "results", "real_data_vectors"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verify-cuda", action="store_true")
    parser.add_argument(
        "--mismatch-summary-only",
        action="store_true",
        help="Build mismatch_summary.csv from an existing summary.csv without inference.",
    )
    parser.add_argument("--force-rerun", "--force_rerun", action="store_true")
    args = parser.parse_args(argv)
    args.bit_widths = _parse_int_csv(args.bit_widths, name="--bit-widths", minimum=4)
    args.bits_to_take_values = _parse_int_csv(
        args.bits_to_take,
        name="--bits-to-take",
        minimum=0,
    )
    args.bits_to_take = 0
    if args.limit_batches < 1:
        raise ValueError("--limit-batches must be at least 1")
    if args.chunk_size != 128:
        raise ValueError("The CUDA dynamic-input search currently requires --chunk-size 128")
    if args.analysis_chunks_per_batch < 1:
        raise ValueError("--analysis-chunks-per-batch must be at least 1")
    if args.max_mismatch_vectors < 0 or args.control_vectors < 0:
        raise ValueError("Vector limits must be non-negative")
    return args


def main(argv=None):
    args = get_args(argv)
    if args.mismatch_summary_only:
        artifact_dir = _artifact_dir(args)
        path = rebuild_mismatch_summary(artifact_dir)
        print(path)
        return
    run_real_data_vectors(args)


if __name__ == "__main__":
    main()
