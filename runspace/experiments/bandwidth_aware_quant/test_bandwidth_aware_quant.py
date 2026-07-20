import os
import sys
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
from torch import nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_pseudo_mse_candidates_force_e1e2_even_without_global_exponent_policy():
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq
    from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

    old_global = getattr(DynamicInputQuantizer, "_global_activation_exponents", None)
    had_global = hasattr(DynamicInputQuantizer, "_global_activation_exponents")
    DynamicInputQuantizer._global_activation_exponents = "all"
    try:
        quantizer = DynamicInputQuantizer(
            nn.Sequential(),
            metric=bwq.PSEUDO_MSE_METRIC,
            candidate_formats=["fp8_e1m6", "fp8_e2m5"],
            collect_error_stats=False,
            collect_format_stats=False,
        )

        assert quantizer._candidates_for_layer("features.0.0") == [
            "fp8_e1m6",
            "fp8_e2m5",
        ]

        quantizer._mark_unsigned_input("features.1.block.1.0", 0)
        assert quantizer._candidates_for_layer("features.1.block.1.0") == [
            "ufp8_e1m7",
            "ufp8_e2m6",
        ]
    finally:
        if had_global:
            DynamicInputQuantizer._global_activation_exponents = old_global
        else:
            delattr(DynamicInputQuantizer, "_global_activation_exponents")


def test_bandwidth_candidates_accept_authoritative_producer_signedness():
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq
    from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 2, bias=False),
    ).eval()
    model[1].input_q_type = "ufp3_e1m2"
    quantizer = DynamicInputQuantizer(
        model,
        metric=bwq.DEFAULT_ACTIVATION_METRIC,
        candidate_formats=["fp3_e1m1", "fp3_e2m0"],
        unsigned_input_sources=["relu"],
        transport="reference",
        collect_error_stats=False,
        collect_format_stats=False,
    )
    quantizer.cache_sim_map = {"1": False}
    quantizer.layer_input_bits_map = {"1": 3}

    assert "1" in quantizer.post_unsigned_layers
    quantizer.register_hooks()
    producer_stage = quantizer._transport_runtime.plan.stage_for_node("_0")

    assert not producer_stage.is_unsigned
    assert quantizer._producer_candidate_cache[producer_stage.stage_id]
    assert all(
        not candidate.startswith("u")
        for candidate in quantizer._producer_candidate_cache[producer_stage.stage_id]
    )
    quantizer.cleanup()


def test_shared_producer_policy_uses_narrowest_width_only():
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq

    fp8 = ("fp8_e4m3", "fp8_e5m2")
    fp3 = ("fp3_e1m1", "fp3_e2m0")

    assert bwq.resolve_shared_producer_policy([fp8, fp3]) == fp3
    assert bwq.resolve_shared_producer_policy([fp3, ("fp3_e2m0",)]) is None
    assert bwq.resolve_shared_producer_policy([fp3, None]) is None


def test_bandwidth_quantizer_reconciles_residual_fanout_widths():
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq
    from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

    class ResidualFanout(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.norm = nn.LayerNorm(4)

        def forward(self, value):
            produced = self.proj(value)
            return self.norm(produced) + produced

    quantizer = DynamicInputQuantizer(
        ResidualFanout().eval(),
        metric=bwq.DEFAULT_ACTIVATION_METRIC,
        candidate_formats=['fp3_e1m1', 'fp3_e2m0'],
        transport='reference',
        collect_error_stats=False,
        collect_format_stats=False,
    )
    quantizer.cache_sim_map = {'norm': True, 'add': False}
    quantizer.layer_input_bits_map = {'norm': 8, 'add': 3}
    quantizer.layer_residual_input_bits_map = {'add': 3}

    quantizer.register_hooks()
    producer_stage = quantizer._transport_runtime.plan.stage_for_node('proj')

    assert quantizer._producer_candidate_cache[producer_stage.stage_id] == [
        'fp3_e1m1',
        'fp3_e2m0',
    ]
    quantizer.cleanup()


def test_legacy_resume_ignores_fake_zero_accuracy_failures(tmp_path):
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq

    signature = {
        'model_name': 'vit_b_16',
        'adapter_type': 'generic',
        'model_source': None,
        'dataset_name': 'imagenet',
        'input_format_policy': 'all',
        'activation_exponents': 'all',
        'activation_metric': 'l2',
        'descent': False,
        'use_best_weights': False,
    }
    saved = {
        'model_name': 'vit_b_16',
        'experiment_config': {
            'adapter_type': 'generic',
            'model_source': None,
            'dataset_name': 'imagenet',
            'input_format_policy': 'all',
            'activation_exponents': 'all',
            'activation_metric': 'mse',
            'descent': False,
            'use_best_weights': False,
        },
        'min_bits_sweeps': {
            '3': {'0.0': [
                {'b': 3, 'accuracy': 0.0, 'cycles': 10},
                {'b': 4, 'accuracy': 72.5, 'cycles': 20},
            ]},
        },
    }
    results_path = tmp_path / 'results.json'
    results_path.write_text(json.dumps(saved))
    progress = {'run_signature': signature, 'points': {}, 'descent': {}}

    bwq.import_completed_results(progress, results_path, signature)

    assert bwq._point_key(0, 3) not in progress['points']
    assert progress['points'][bwq._point_key(0, 4)]['acc1'] == 72.5


def test_descent_does_not_choose_winner_when_any_candidate_failed(monkeypatch, tmp_path):
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq

    monkeypatch.setattr(bwq, 'lowest_activation_bit_width', lambda _args: 8)
    monkeypatch.setattr(bwq, 'SIGNED_FORMATS_BY_BITS', {8: ['fp8_e4m3']})
    monkeypatch.setattr(
        bwq,
        'compute_model_runtime',
        lambda *_args, **_kwargs: (10, {}, {'layer': 8}, {}, {}, {}),
    )
    monkeypatch.setattr(
        bwq,
        'create_descent_state_dict',
        lambda *_args, **_kwargs: ({}, {'layer': 'fp8_e4m3'}),
    )
    monkeypatch.setattr(bwq, 'build_eval_config', lambda *_args, **_kwargs: {})

    class Runner:
        calls = 0

        def run_single(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {'acc1': 75.0}
            raise RuntimeError('synthetic evaluation failure')

    args = SimpleNamespace(
        bandwidth=1.0,
        input_format_policy='all',
        activation_exponents='all',
    )
    resume_data = {}

    with pytest.raises(RuntimeError, match='failed candidates'):
        bwq.run_descent_for_cache(
            model=None,
            model_name='toy',
            args=args,
            runner=Runner(),
            sim_layers=[],
            cache_sim_map={},
            model_output_dir=str(tmp_path),
            temp_weights_dir=str(tmp_path),
            resume_data=resume_data,
        )

    assert resume_data['candidates']['8']['fp8_e4m3']['acc1'] == 75.0
    assert resume_data['levels'] == {}


def test_pseudo_mse_descent_uses_distinct_results_dir_and_normalized_flags():
    from runspace.experiments.bandwidth_aware_quant import bandwidth_aware_quant as bwq

    args = Namespace(
        pseudo_mse_descent=True,
        pseudo_mse=False,
        descent=False,
        activation_metric="mse",
        activation_exponents="all",
        b_bits=None,
    )

    class Parser:
        def error(self, message):
            raise AssertionError(message)

    bwq.configure_activation_metric_args(Parser(), args)

    assert args.pseudo_mse is True
    assert args.descent is True
    assert args.activation_metric == bwq.PSEUDO_MSE_METRIC
    assert args.activation_exponents == "e1e2"

    result_args = SimpleNamespace(
        descent=args.descent,
        use_best_weights=False,
        activation_metric=args.activation_metric,
        activation_exponents=args.activation_exponents,
    )
    assert bwq.default_results_dir_for_args(result_args).endswith(
        "results_descent_pseudo_mse_activation_e1e2"
    )


def test_cache_sizes_always_include_a_unique_fp32_zero_cache_reference():
    from runspace.experiments.bandwidth_aware_quant import runtime_model

    assert runtime_model.cache_sizes_with_fp32_reference([2.0, 4.0]) == [0.0, 2.0, 4.0]
    assert runtime_model.cache_sizes_with_fp32_reference([0.0, 2.0, 0.0]) == [0.0, 2.0]


def test_fp32_and_quantized_runtime_share_residual_transfer_accounting():
    from runspace.experiments.bandwidth_aware_quant import runtime_model

    layers = [
        {
            "name": "add",
            "type": "QuantAdd",
            "input_elems": 128,
            "weight_elems": 0,
            "output_elems": 128,
            "input_shapes": [(1, 128), (1, 128)],
            "stay_on_chip": True,
            "xin_from_cache": True,
            "residual_input_stream_elems": 128,
        }
    ]

    quantized = runtime_model.compute_model_runtime(layers, 4, bandwidth=1.0)
    fp32 = runtime_model.compute_model_runtime(layers, 32, bandwidth=1.0, max_bits=32)

    # Each runtime transfers two 128-element inputs: the main input and one
    # residual operand. At 1 byte/cycle, that is 2 * 16 * bits cycles.
    assert quantized[0] == 2 * 16 * 4
    assert fp32[0] == 2 * 16 * 32
    assert fp32[0] / quantized[0] == 8.0
    assert quantized[4] == {"add": 4}
    assert fp32[4] == {"add": 32}
