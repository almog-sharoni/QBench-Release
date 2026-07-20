import json

from runspace.src.database.bwaware_results import (
    is_greedy_descent_result,
    load_greedy_descent_results,
    normalize_cache_cycles,
)


def _write_result(path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_cache_cycles_treats_integer_and_float_keys_equally():
    cycles = normalize_cache_cycles({"0.0": 300.0, "1": 200.0, "2": 150.0})

    assert cycles == {0.0: 300.0, 1.0: 200.0, 2.0: 150.0}
    assert cycles[2.0] == 150.0


def test_is_greedy_descent_result_accepts_current_and_legacy_metadata():
    assert is_greedy_descent_result({"used_descent": True})
    assert is_greedy_descent_result({"experiment_config": {"descent": True}})
    assert not is_greedy_descent_result(
        {"experiment_config": {"descent": False, "use_best_weights": True}}
    )


def test_loader_discovers_all_result_trees_and_returns_only_greedy(tmp_path):
    base = tmp_path / "runspace/experiments/bandwidth_aware_quant"
    filename = "bandwidth_aware_quant_results.json"
    _write_result(
        base / "results_descent" / "resnet50" / filename,
        model_name="resnet50",
        used_descent=True,
    )
    _write_result(
        base / "results_descent_activation_e1e2" / "vit_b_16" / filename,
        model_name="vit_b_16",
        experiment_config={"descent": True},
    )
    _write_result(
        base / "results_best_weights" / "mobilenet" / filename,
        model_name="mobilenet",
        experiment_config={"descent": False, "use_best_weights": True},
    )

    runs = load_greedy_descent_results(str(tmp_path))

    assert [run["model_name"] for run in runs] == ["resnet50", "vit_b_16"]
    assert all("results_descent" in run["label"] for run in runs)
