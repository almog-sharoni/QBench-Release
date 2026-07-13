import csv
import json
import os

import pytest
import torch
import torch.nn as nn

from runspace.core.runner import Runner
from runspace.experiments.pseudo_mse3.generate_real_data_vectors import (
    BaseChunkAnalysis,
    RealActivationCollector,
    StatsBucket,
    _analyze_chunks_reference,
    analyze_chunks,
    fixed_analysis_from_diff,
    fixed_analyses_from_diff,
    get_args,
    rebuild_mismatch_summary,
    run_real_data_vectors,
    validate_vectors_csv,
    write_outputs,
)
from runspace.experiments.pseudo_mse3.plot_format_choices import (
    format_choice_counts_by_bit_width,
    plot_format_choice_counts_from_rows,
)
from runspace.src.quantization.dynamic_input_metrics import (
    validate_pseudo_mse_candidate_pairs,
)
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer


CANDIDATES = ("fp8_e1m6", "fp8_e2m5")


def _indices_from_choice(choice, candidates=CANDIDATES):
    pair = validate_pseudo_mse_candidate_pairs(candidates)[0]
    return torch.where(
        choice,
        torch.full_like(choice, pair.e2_index, dtype=torch.long),
        torch.full_like(choice, pair.e1_index, dtype=torch.long),
    )


def test_fixed_analysis_exposes_per_element_floor_bias():
    exact_diff = torch.tensor([[0.3, -0.2]], dtype=torch.float32)

    exact = fixed_analysis_from_diff(exact_diff, bits_to_take=0)
    fixed = fixed_analysis_from_diff(exact_diff, bits_to_take=1)
    nearest = fixed_analysis_from_diff(
        exact_diff,
        bits_to_take=1,
        fixed_rounding="nearest",
    )

    assert exact.chunk_sum.item() > 0
    assert not exact.choose_exp2.item()
    assert fixed.contributions.tolist() == [[0, -1]]
    assert fixed.chunk_sum.item() == -1
    assert fixed.choose_exp2.item()
    assert nearest.contributions.tolist() == [[4, -2]]
    assert nearest.chunk_sum.item() == 2
    assert not nearest.choose_exp2.item()
    fixed_zero_exp1 = fixed_analysis_from_diff(
        torch.tensor([[0.3, 0.2]], dtype=torch.float32),
        bits_to_take=1,
        fixed_rounding="floor",
        tie_break="exp1",
    )
    fixed_zero_exp2 = fixed_analysis_from_diff(
        torch.tensor([[0.3, 0.2]], dtype=torch.float32),
        bits_to_take=1,
        fixed_rounding="floor",
        tie_break="exp2",
    )
    assert fixed_zero_exp1.chunk_sum.item() == 0
    assert not fixed_zero_exp1.choose_exp2.item()
    assert fixed_zero_exp2.chunk_sum.item() == 0
    assert fixed_zero_exp2.choose_exp2.item()
    with pytest.raises(OverflowError):
        fixed_analysis_from_diff(torch.ones((1, 1)), bits_to_take=40)


@pytest.mark.parametrize("fixed_rounding", ["floor", "nearest"])
@pytest.mark.parametrize("tie_break", ["exp1", "exp2"])
def test_batched_fixed_analysis_and_stats_are_exactly_scalar_equivalent(
    fixed_rounding,
    tie_break,
):
    generator = torch.Generator().manual_seed(2026)
    raw_chunks = torch.randn((17, 128), generator=generator, dtype=torch.float32)
    base = analyze_chunks(raw_chunks, CANDIDATES)
    bits_values = (0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24)
    batched = fixed_analyses_from_diff(
        base.pseudo_diff,
        bits_values,
        fixed_rounding=fixed_rounding,
        tie_break=tie_break,
    )
    runtime_choice = base.reference_choose_exp2.clone()
    runtime_choice[::3] = ~runtime_choice[::3]
    count, batched_counts, batched_aggregates = StatsBucket.compute_deltas(
        base,
        batched,
        runtime_choice,
        bits_values,
    )

    assert count == raw_chunks.shape[0]
    for index, bits_to_take in enumerate(bits_values):
        scalar = fixed_analysis_from_diff(
            base.pseudo_diff,
            bits_to_take,
            fixed_rounding=fixed_rounding,
            tie_break=tie_break,
        )
        assert torch.equal(batched[bits_to_take].contributions, scalar.contributions)
        assert torch.equal(batched[bits_to_take].chunk_sum, scalar.chunk_sum)
        assert torch.equal(batched[bits_to_take].choose_exp2, scalar.choose_exp2)
        scalar_count, scalar_counts, scalar_aggregates = StatsBucket.compute_delta(
            base,
            scalar,
            runtime_choice,
        )
        assert scalar_count == count
        assert torch.equal(batched_counts[index], scalar_counts)
        assert torch.equal(batched_aggregates[index], scalar_aggregates)


@pytest.mark.parametrize(
    "candidates",
    [
        ("fp8_e1m6", "fp8_e2m5"),
        ("ufp8_e1m7", "ufp8_e2m6"),
    ],
)
def test_cuda_analysis_is_bitwise_identical_to_python_reference(candidates):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(2027)
    raw_chunks = torch.randn(31, 128, device="cuda", dtype=torch.float32)
    raw_chunks[0, :8] = torch.tensor(
        [0.0, -0.0, 2.0**-126, -(2.0**-126), 0.5, -0.5, 1.0, -1.0],
        device="cuda",
    )

    fast = analyze_chunks(raw_chunks, candidates)
    reference = _analyze_chunks_reference(raw_chunks, candidates)

    for field_name in BaseChunkAnalysis.__dataclass_fields__:
        fast_value = getattr(fast, field_name)
        reference_value = getattr(reference, field_name)
        if isinstance(fast_value, torch.Tensor):
            assert torch.equal(fast_value, reference_value), field_name
        else:
            assert fast_value == reference_value


def test_runner_preserves_chunk_observer_identity():
    def observer(**_kwargs):
        return None

    runner = object.__new__(Runner)

    normalized = runner._normalize_input_quant_cfg(
        input_quant_cfg={
            "enabled": True,
            "mode": "dynamic",
            "metric": "mse",
            "chunk_observer": observer,
        }
    )

    assert normalized["chunk_observer"] is observer

    normalized_legacy = runner._normalize_input_quant_cfg(
        input_quant_cfg={"chunk_observer": observer},
        dynamic_input_quant_cfg={"enabled": True, "metric": "mse"},
    )
    assert normalized_legacy["chunk_observer"] is observer


def test_dynamic_quantizer_emits_selected_chunk_batch():
    seen = []

    def observer(**observation):
        seen.append(observation)

    quantizer = DynamicInputQuantizer(
        model=nn.Identity(),
        metric="pseudo_mse3",
        chunk_size=128,
        candidate_formats=CANDIDATES,
        collect_error_stats=False,
        collect_format_stats=False,
        chunk_observer=observer,
    )
    values = torch.linspace(-1.75, 1.75, 256, dtype=torch.float32).reshape(2, 128)

    _quantized, best_indices = quantizer._select_best_format(
        values,
        "test.layer",
        CANDIDATES,
    )

    assert len(seen) == 1
    assert seen[0]["layer_name"] == "test.layer"
    assert seen[0]["candidates"] == CANDIDATES
    torch.testing.assert_close(seen[0]["ref_chunks"], values)
    assert torch.equal(seen[0]["best_indices"], best_indices)


def test_real_activation_collector_counts_and_exports_round_trip(tmp_path):
    generator = torch.Generator().manual_seed(42)
    raw_chunks = torch.randn((12, 128), generator=generator, dtype=torch.float32)
    base = analyze_chunks(raw_chunks, CANDIDATES)
    fixed = fixed_analysis_from_diff(base.pseudo_diff, bits_to_take=1)
    runtime_choice = fixed.choose_exp2.clone()
    runtime_choice[:6] = ~runtime_choice[:6]
    runtime_indices = _indices_from_choice(runtime_choice)
    collector = RealActivationCollector(
        bits_to_take=(1,),
        analysis_chunks_per_batch=3,
        max_mismatch_vectors=2,
        control_vectors=1,
        seed=42,
    )

    collector.observe(
        layer_name="features.0",
        candidates=CANDIDATES,
        ref_chunks=raw_chunks,
        best_indices=runtime_indices,
    )

    summary_rows = collector.summary_rows()
    global_row = next(row for row in summary_rows if row["scope"] == "global")
    layer_row = next(row for row in summary_rows if row["scope"] == "layer")
    assert global_row["total_chunks"] == 12
    assert global_row["fixed_rounding"] == "floor"
    assert global_row["tie_break"] == "exp1"
    assert global_row["decision_mismatches"] == 6
    assert layer_row["total_chunks"] == global_row["total_chunks"]
    samples = collector.samples()
    assert sum(sample.category == "mismatch" for sample in samples) == 2
    assert sum(sample.category == "control" for sample in samples) == 1
    full_mismatch = fixed.choose_exp2 != runtime_choice
    ordinals = torch.arange(raw_chunks.shape[0], dtype=torch.int64)
    for category, mask, capacity in (
        ("mismatch", full_mismatch, 2),
        ("control", ~full_mismatch, 1),
    ):
        priorities = collector._priorities(ordinals, 1, category)
        expected = sorted(
            (
                (int(priorities[index]), int(index))
                for index in torch.nonzero(mask, as_tuple=False).flatten()
            ),
            key=lambda item: item[0],
        )[:capacity]
        actual = sorted(
            (
                (sample.priority, sample.global_chunk_index)
                for sample in samples
                if sample.category == category
            ),
            key=lambda item: item[0],
        )
        assert actual == expected
        for sample in samples:
            if sample.category == category:
                assert torch.equal(sample.raw_chunk, raw_chunks[sample.global_chunk_index])

    paths = write_outputs(
        str(tmp_path),
        collector,
        model_name="test_model",
        manifest={"test": True},
    )

    assert validate_vectors_csv(paths["vectors_csv"]) == 3
    assert os.path.getsize(paths["format_choices_plot"]) > 0
    with open(paths["mismatch_summary"], newline="") as handle:
        mismatch_rows = list(csv.DictReader(handle))
    assert len(mismatch_rows) == 1
    assert int(mismatch_rows[0]["bit_width"]) == 8
    assert int(mismatch_rows[0]["bits_to_take"]) == 1
    assert mismatch_rows[0]["fixed_rounding"] == "floor"
    assert mismatch_rows[0]["tie_break"] == "exp1"
    assert int(mismatch_rows[0]["total_chunks"]) == 12
    assert int(mismatch_rows[0]["matched_chunks"]) == 6
    assert int(mismatch_rows[0]["decision_mismatches"]) == 6
    assert float(mismatch_rows[0]["mismatch_percent"]) == 50.0
    with open(paths["vectors_csv"], newline="") as handle:
        vector_rows = list(csv.DictReader(handle))
    assert len(vector_rows) == 3 * 128
    assert "BEGIN_VECTOR" in open(paths["vectors_txt"]).read()
    manifest = json.loads(open(paths["manifest"]).read())
    assert manifest["validated_vectors"] == 3
    assert manifest["fixed_rounding"] == "floor"
    assert manifest["tie_break"] == "exp1"
    assert manifest["summary_rows"] == 2
    assert manifest["mismatch_summary_rows"] == 1
    assert manifest["artifacts"]["mismatch_summary_csv"] == "mismatch_summary.csv"
    assert (
        manifest["artifacts"]["format_choices_plot"]
        == "format_choices_mse_vs_pseudo_mse3.png"
    )

    with open(paths["summary"], newline="") as handle:
        detailed_reader = csv.DictReader(handle)
        detailed_rows = list(detailed_reader)
        detailed_fields = list(detailed_reader.fieldnames)
    padded_fields = [f" {field} " for field in detailed_fields]
    with open(paths["summary"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=padded_fields)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(
                {
                    padded: f" {row[original]} "
                    for original, padded in zip(detailed_fields, padded_fields)
                }
            )
    os.remove(paths["mismatch_summary"])
    os.remove(paths["format_choices_plot"])
    assert rebuild_mismatch_summary(str(tmp_path)) == paths["mismatch_summary"]
    assert os.path.exists(paths["mismatch_summary"])
    assert os.path.exists(paths["format_choices_plot"])


def test_format_choice_plot_groups_mse_and_each_bits_to_take(tmp_path):
    rows = [
        {
            " bit_width ": f" {bit_width} ",
            " bits_to_take ": f" {bits_to_take} ",
            " fixed_rounding ": " nearest ",
            " tie_break ": " exp1 ",
            " runtime_mse_e1 ": " 70 ",
            " runtime_mse_e2 ": " 30 ",
            " pseudo_e1 ": f" {pseudo_e1} ",
            " pseudo_e2 ": f" {100 - pseudo_e1} ",
        }
        for bit_width, bits_to_take, pseudo_e1 in (
            (8, 0, 70),
            (8, 13, 64),
            (7, 0, 70),
            (7, 13, 61),
        )
    ]

    grouped = format_choice_counts_by_bit_width(rows)

    assert list(grouped) == [7, 8]
    assert grouped[8]["mse"] == {"e1": 70, "e2": 30}
    assert grouped[8]["pseudo"][0] == {"e1": 70, "e2": 30}
    assert grouped[8]["pseudo"][13] == {"e1": 64, "e2": 36}

    plot_path = tmp_path / "format_choices.png"
    result = plot_format_choice_counts_from_rows(rows, str(plot_path), dpi=80)

    assert result == str(plot_path)
    assert plot_path.stat().st_size > 0


def test_real_vector_cli_defaults_and_requested_sweep():
    args = get_args([])

    assert args.bit_widths == [8, 7, 6, 5, 4]
    assert args.bits_to_take_values == [0, 1, 3, 5, 7, 9]
    assert args.bits_to_take == 0
    assert args.fixed_rounding == "floor"
    assert args.tie_break == "exp1"
    assert args.limit_batches == 1

    nearest_args = get_args(["--fixed-rounding", "rtn"])
    assert nearest_args.fixed_rounding == "nearest"
    tie_args = get_args(["--tie-break", "le"])
    assert tie_args.tie_break == "exp2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fixed_point_preserves_equal_reconstruction_ties():
    from runspace.src.quantization.cuda import search_best_chunk_format

    candidates = CANDIDATES
    values = torch.linspace(-1.99, 1.99, 8192, dtype=torch.float32).reshape(64, 128)
    base = analyze_chunks(values, candidates)
    common = (base.q1_scaled == base.q2_scaled) & (base.err1_pre_square != 0)
    common_values = base.scaled_chunks[common][:128]
    assert common_values.numel() == 128
    raw_chunk = common_values.reshape(1, 128).cuda()
    tie_base = analyze_chunks(raw_chunk.cpu(), candidates)
    assert torch.count_nonzero(tie_base.exact_diff).item() == 0

    best_indices, _scales, _quantized, _unscaled = search_best_chunk_format(
        raw_chunk.reshape(-1).contiguous(),
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        torch.tensor([6, 5], dtype=torch.int32, device="cuda"),
        torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        False,
        9,
        1.0,
        0,
    )

    assert best_indices.item() == 0

    tie_exp2_indices, _scales, _quantized, _unscaled = search_best_chunk_format(
        raw_chunk.reshape(-1).contiguous(),
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        torch.tensor([6, 5], dtype=torch.int32, device="cuda"),
        torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        False,
        9,
        1.0,
        0,
        0,
        1,
    )
    assert tie_exp2_indices.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_activation_style_nearest_matches_python_reference():
    from runspace.src.quantization.cuda import search_best_chunk_format

    generator = torch.Generator().manual_seed(123)
    raw_cpu = torch.randn((512, 128), generator=generator, dtype=torch.float32)
    base = analyze_chunks(raw_cpu, CANDIDATES)
    raw_cuda = raw_cpu.cuda()
    pair = validate_pseudo_mse_candidate_pairs(CANDIDATES)[0]
    bits_values = (12, 16, 20)
    fixed_analyses = fixed_analyses_from_diff(
        base.pseudo_diff,
        bits_values,
        fixed_rounding="nearest",
    )

    for bits_to_take in bits_values:
        fixed = fixed_analyses[bits_to_take]
        expected_indices = _indices_from_choice(fixed.choose_exp2)
        best_indices, _scales, _quantized, _unscaled = search_best_chunk_format(
            raw_cuda.reshape(-1).contiguous(),
            torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([6, 5], dtype=torch.int32, device="cuda"),
            torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
            False,
            9,
            float(bits_to_take),
            0,
            1,
        )
        assert pair.e1_index == 0
        assert pair.e2_index == 1
        assert torch.equal(best_indices.cpu(), expected_indices)


@pytest.mark.skipif(
    os.environ.get("QBENCH_RUN_IMAGENET_TESTS") != "1"
    or not torch.cuda.is_available()
    or not os.path.isdir("/data/imagenet/val"),
    reason="Set QBENCH_RUN_IMAGENET_TESTS=1 with CUDA and ImageNet mounted",
)
def test_real_activation_resnet50_imagenet_smoke(tmp_path):
    args = get_args(
        [
            "--bit-widths",
            "8",
            "--bits-to-take",
            "0,1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--limit-batches",
            "1",
            "--max-mismatch-vectors",
            "1",
            "--control-vectors",
            "1",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cuda",
        ]
    )

    paths = run_real_data_vectors(args, device=torch.device("cuda"))

    assert all(os.path.exists(path) for path in paths.values())
    manifest = json.loads(open(paths["manifest"]).read())
    assert manifest["observer_calls"] > 0
    assert manifest["bit_widths"] == [8]
