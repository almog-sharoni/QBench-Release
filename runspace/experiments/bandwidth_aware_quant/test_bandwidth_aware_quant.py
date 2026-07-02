import os
import sys
from argparse import Namespace
from types import SimpleNamespace

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
