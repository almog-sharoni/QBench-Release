from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (
    _candidate_formats_by_bit_width,
    _dynamic_experiment_type_for_bit_width,
    _validate_cache_sim_candidate_groups,
    candidate_formats as DEFAULT_CANDIDATE_FORMATS,
)
from runspace.experiments.find_optimal_input_quant.verify_dynamic_input_replay import (
    BatchRecordingDynamicInputQuantizer,
    BatchReplayInputQuantizer,
    StageFormatSelection,
)


class _TinyStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()
        self.proj_out = nn.Linear(4, 2, bias=False)

    def forward(self, tensor):
        return self.proj_out(self.relu(self.proj_in(tensor)))


def _pre_hook_count(model):
    return sum(len(module._forward_pre_hooks) for module in model.modules())


def test_dynamic_defaults_exclude_fp32_hardware_candidate():
    assert "fp32" not in DEFAULT_CANDIDATE_FORMATS


def test_dynamic_candidates_are_split_into_independent_bit_width_pools():
    grouped = _candidate_formats_by_bit_width(DEFAULT_CANDIDATE_FORMATS)

    assert list(grouped) == [8, 7, 6, 5, 4, 3, 2]
    assert grouped[4] == ["fp4_e1m2", "fp4_e2m1", "fp4_e3m0"]
    assert all(
        candidate.startswith(f"fp{bit_width}_")
        for bit_width, candidates in grouped.items()
        for candidate in candidates
    )


def test_dynamic_experiment_type_is_suffixed_with_bit_width():
    assert (
        _dynamic_experiment_type_for_bit_width("input_quant_dynamic", 4)
        == "input_quant_dynamic_4"
    )


def test_cache_sim_rejects_non_8_bit_candidate_runs():
    with pytest.raises(ValueError, match="cannot produce isolated non-8-bit"):
        _validate_cache_sim_candidate_groups({8: ["fp8_e4m3"], 4: ["fp4_e2m1"]}, True)

    _validate_cache_sim_candidate_groups({8: ["fp8_e4m3"]}, True)


def test_dynamic_quantizer_rejects_fp32_candidate():
    with pytest.raises(ValueError, match="fp32 cannot be a dynamic activation candidate"):
        BatchRecordingDynamicInputQuantizer(
            _TinyStageModel(),
            candidate_formats=["fp32", "fp4_e1m2"],
        )


def test_reference_replay_uses_exact_producer_stage_format_ids():
    torch.manual_seed(7)
    capture_model = _TinyStageModel().eval()
    replay_model = copy.deepcopy(capture_model)
    candidates = ["fp4_e1m2", "fp4_e2m1"]

    capture = BatchRecordingDynamicInputQuantizer(
        capture_model,
        metric="mse",
        chunk_size=4,
        candidate_formats=candidates,
        transport="reference",
    )
    replay = BatchReplayInputQuantizer(
        replay_model,
        chunk_size=4,
        candidate_formats=candidates,
        transport="reference",
    )

    capture.register_hooks()
    replay.register_hooks()
    try:
        assert _pre_hook_count(capture_model) == 0
        assert _pre_hook_count(replay_model) == 0

        capture_stage_ids = tuple(
            stage.stage_id for stage in capture._transport_runtime.plan.stages
        )
        replay_stage_ids = tuple(
            stage.stage_id for stage in replay._transport_runtime.plan.stages
        )
        assert replay_stage_ids == capture_stage_ids

        inputs = torch.tensor(
            [[-1.25, -0.5, 0.75, 1.5], [0.125, 0.5, -0.875, 2.0]],
            dtype=torch.float32,
        )
        capture.begin_batch()
        expected = capture_model(inputs)
        batch_plan = capture.consume_batch_plan()

        assert batch_plan
        assert set(batch_plan).issubset(capture_stage_ids)
        unsigned_stage_ids = {
            stage.stage_id
            for stage in capture._transport_runtime.plan.stages
            if stage.is_unsigned
        }
        assert unsigned_stage_ids & set(batch_plan)
        for stage_id in unsigned_stage_ids & set(batch_plan):
            for selection in batch_plan[stage_id]:
                assert all(fmt.startswith("ufp") for fmt in selection.candidate_formats)

        replay.load_batch_plan(copy.deepcopy(batch_plan))
        actual = replay_model(inputs)
        replay.assert_batch_fully_consumed()

        assert torch.equal(actual, expected)
        assert replay.get_final_stats()["transport"] == "reference"
        assert replay.get_final_stats()["transmission_count"] == sum(
            len(selections) for selections in batch_plan.values()
        )
    finally:
        capture.cleanup()
        replay.cleanup()


def test_replay_reports_missing_and_unconsumed_stage_records():
    replay = BatchReplayInputQuantizer(
        _TinyStageModel(),
        chunk_size=4,
        candidate_formats=["fp4_e1m2"],
        transport="reference",
    )

    with pytest.raises(RuntimeError, match="No replay plan found for producer stage"):
        replay._consume_stage_selection("stage_missing")

    replay.load_batch_plan(
        {
            "stage_input": [
                StageFormatSelection(
                    candidate_formats=("fp4_e1m2",),
                    format_ids=b"\x00",
                )
            ]
        }
    )
    with pytest.raises(RuntimeError, match="did not consume all recorded producer"):
        replay.assert_batch_fully_consumed()


def test_replay_defaults_to_encoded_transport():
    capture = BatchRecordingDynamicInputQuantizer(
        _TinyStageModel(),
        candidate_formats=["fp4_e1m2"],
    )
    replay = BatchReplayInputQuantizer(
        _TinyStageModel(),
        candidate_formats=["fp4_e1m2"],
    )

    assert capture.transport == "encoded"
    assert replay.transport == "encoded"
