#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MARKER = "PHASE3_JSON="
PHASES = ("initial_residency", "kernel_execution", "result_readback")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def load_case_matrix(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_result(path):
    for line in path.read_text().splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise ValueError(f"No {MARKER} record in {path}")


def flatten_measurement(split, case_id, repeat, data, stdout_path, stderr_path, exit_code):
    phases = data["phases"]
    row = {
        "split": split,
        "case_id": case_id,
        "repeat": repeat,
        "kernel": data["kernel"],
        "evidence_label": data["evidence_label"],
        "logical_elements": data.get("logical_elements"),
        "padded_elements": data.get("padded_elements"),
        "logical_output_dim": data.get("logical_output_dim"),
        "logical_input_dim": data.get("logical_input_dim"),
        "vectors": data.get("vectors"),
        "parallel_tiles": data.get("parallel_tiles"),
        "input_tiles": data.get("input_tiles"),
        "input_count": data.get("input_count"),
        "padded_bursts": data.get("padded_bursts"),
        "weight_bursts": data.get("weight_bursts"),
        "partial_sum_bursts": data.get("partial_sum_bursts"),
        "initial_residency_cycles": phases["initial_residency"]["cycles"],
        "kernel_execution_cycles": phases["kernel_execution"]["cycles"],
        "result_readback_cycles": phases["result_readback"]["cycles"],
        "native_total_cycles": phases["native_total"]["cycles"],
        "native_total_read_transactions": phases["native_total"]["read_transactions"],
        "native_total_write_transactions": phases["native_total"]["write_transactions"],
        "exact_bit_mismatches": data["exact_bit_mismatches"],
        "outside_pim_uncosted_operations": data["outside_pim_uncosted_operations"],
        "outside_pim_uncosted_operation_type": data.get("outside_pim_uncosted_operation_type"),
        "exit_code": exit_code,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }
    traffic = data.get("traffic_bytes", {})
    for key, value in traffic.items():
        row[f"traffic_{key}_bytes"] = value
    return row


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fit_ols(rows, kernel, phase):
    selected = [row for row in rows if row["kernel"] == kernel]
    if kernel in {"ADD", "RELU"}:
        if phase == "initial_residency":
            features = ["constant", "input_count_times_padded_bursts"]
            matrix = [[1.0, row["input_count"] * row["padded_bursts"]] for row in selected]
        elif phase == "kernel_execution":
            features = ["constant", "parallel_tiles"]
            matrix = [[1.0, row["parallel_tiles"]] for row in selected]
        else:
            features = ["constant", "padded_bursts"]
            matrix = [[1.0, row["padded_bursts"]] for row in selected]
    elif kernel == "GEMV":
        if phase == "initial_residency":
            features = ["constant", "weight_bursts"]
            matrix = [[1.0, row["weight_bursts"]] for row in selected]
        elif phase == "kernel_execution":
            features = ["constant", "input_tiles", "vectors", "input_tiles_times_vectors"]
            matrix = [[1.0, row["input_tiles"], row["vectors"],
                       row["input_tiles"] * row["vectors"]] for row in selected]
        else:
            features = ["constant", "partial_sum_bursts"]
            matrix = [[1.0, row["partial_sum_bursts"]] for row in selected]
    else:
        raise ValueError(kernel)
    target = np.asarray([row[f"{phase}_cycles"] for row in selected], dtype=float)
    design = np.asarray(matrix, dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    return {
        "kernel": kernel,
        "phase": phase,
        "features": features,
        "coefficients": coefficients.tolist(),
        "rank": int(rank),
        "observations": len(selected),
        "singular_values": singular_values.tolist(),
    }


def predict(model, row):
    values = {
        "constant": 1.0,
        "input_count_times_padded_bursts": (row.get("input_count") or 0) * (row.get("padded_bursts") or 0),
        "parallel_tiles": row.get("parallel_tiles") or 0,
        "padded_bursts": row.get("padded_bursts") or 0,
        "weight_bursts": row.get("weight_bursts") or 0,
        "input_tiles": row.get("input_tiles") or 0,
        "vectors": row.get("vectors") or 0,
        "input_tiles_times_vectors": (row.get("input_tiles") or 0) * (row.get("vectors") or 0),
        "partial_sum_bursts": row.get("partial_sum_bursts") or 0,
    }
    return sum(coef * values[name]
               for coef, name in zip(model["coefficients"], model["features"]))


def summarize_traces(run_dir, validation_rows, output_path):
    command_pattern = re.compile(r"^(READ|WRITE|ACTIVATE|PRECHARGE) .* @(\d+)(?: tag : (.*))?$")
    summary = []
    for row in validation_rows:
        case_id = row["case_id"]
        trace_path = run_dir / "traces" / case_id / "stdout.txt.gz"
        counts = Counter()
        tags = Counter()
        max_cycle = 0
        line_count = 0
        uncompressed_bytes = 0
        marker_found = False
        with gzip.open(trace_path, "rt", errors="replace") as handle:
            for line in handle:
                line_count += 1
                uncompressed_bytes += len(line.encode("utf-8"))
                if line.startswith(MARKER):
                    marker_found = True
                match = command_pattern.match(line.rstrip("\n"))
                if not match:
                    continue
                command, cycle, tag = match.groups()
                counts[command] += 1
                max_cycle = max(max_cycle, int(cycle))
                if tag:
                    tags[tag] += 1
        summary.append({
            "case_id": case_id,
            "kernel": row["kernel"],
            "trace_sha256": sha256(trace_path),
            "compressed_bytes": trace_path.stat().st_size,
            "uncompressed_bytes": uncompressed_bytes,
            "line_count": line_count,
            "command_lines": sum(counts.values()),
            "read_commands": counts["READ"],
            "write_commands": counts["WRITE"],
            "activate_commands": counts["ACTIVATE"],
            "precharge_commands": counts["PRECHARGE"],
            "tagged_park_commands": sum(value for tag, value in tags.items() if "PARK" in tag),
            "tagged_program_crf_commands": sum(value for tag, value in tags.items() if "PROGRAM_CRF" in tag),
            "tagged_compute_commands": sum(value for tag, value in tags.items()
                                                   if any(token in tag for token in ("MAC", "ADD", "ReLU"))),
            "maximum_command_cycle": max_cycle,
            "result_marker_found": marker_found,
        })
    write_csv(output_path, summary)
    return summary


def quality_accounting(arm_dir, output_path):
    project_root = arm_dir.parents[2]
    quality_path = project_root / "runspace/experiments/find_optimal_hybrid_quant/results/latest_db_results.json"
    records = json.loads(quality_path.read_text())
    model = next(record for record in records if record.get("model_name") == "resnet50")
    rows = model["hybrid_results"]
    run_ids = [row["run_id"] for row in rows]
    duplicate_groups = {}
    for row in rows:
        duplicate_groups.setdefault(str(row["run_id"]), []).append({
            "selection_direction": row.get("selection_direction"),
            "reconstructed_duplicate": row.get("reconstructed_duplicate", False),
            "weight_type": row.get("weight_type"),
            "input_type": row.get("input_type"),
        })
    duplicate_groups = {
        run_id: group for run_id, group in duplicate_groups.items() if len(group) > 1
    }
    output = {
        "model_name": "resnet50",
        "source_path": str(quality_path.relative_to(project_root)),
        "source_sha256": sha256(quality_path),
        "directional_record_count": len(rows),
        "unique_physical_run_id_count": len(set(run_ids)),
        "reconstructed_directional_record_count": sum(bool(row.get("reconstructed_duplicate")) for row in rows),
        "multi_view_run_ids": duplicate_groups,
        "statistical_sample_count_used_by_phase3": 0,
        "pim_cost_assignment_from_quality_records": False,
        "reason": "All sub-16-bit representations are unsupported by the locked native FP16 PIM mapping. Directional duplicates remain visible but are not measurements consumed by this arm."
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def make_plots(arm_dir, validation_rows, validation_table):
    figures = arm_dir / "figures"
    figures.mkdir(exist_ok=True)
    labels = [row["case_id"].replace("_", "\n") for row in validation_rows]
    residency = [row["initial_residency_cycles"] for row in validation_rows]
    execution = [row["kernel_execution_cycles"] for row in validation_rows]
    readback = [row["result_readback_cycles"] for row in validation_rows]
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(positions, residency, label="Initial residency", color="#4C78A8")
    ax.bar(positions, execution, bottom=residency, label="Kernel execution", color="#F58518")
    bottoms = np.asarray(residency) + np.asarray(execution)
    ax.bar(positions, readback, bottom=bottoms, label="Result readback", color="#54A24B")
    ax.set_yscale("log")
    ax.set_ylabel("Simulator cycles (log scale, tCK = 1 ns)")
    ax.set_xticks(positions, labels)
    ax.set_title("ResNet-50 representative native FP16 HBM2-PIM costs")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "workload_cycle_breakdown.png", dpi=180)
    plt.close(fig)

    errors = [100.0 * row["absolute_relative_error"] for row in validation_table]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#54A24B" if error <= 20.0 else "#E45756" for error in errors]
    ax.bar(np.arange(len(errors)), errors, color=colors)
    ax.axhline(20.0, color="#E45756", linestyle="--", linewidth=1.5, label="Per-case gate (20%)")
    ax.set_ylabel("Absolute relative error (%)")
    ax.set_xticks(np.arange(len(errors)), labels)
    ax.set_title("Locked analytical-vs-simulator validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "locked_validation_error.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    arm_dir = args.arm_dir.resolve()
    run_dir = arm_dir / "raw" / args.run_id
    case_rows = load_case_matrix(arm_dir / "case_matrix.csv")

    measurements = []
    parsed_by_case = {}
    adapter_pass = True
    determinism_pass = True
    for case in case_rows:
        repeat_data = []
        for repeat in (1, 2):
            result_dir = run_dir / "results" / case["split"] / case["case_id"] / f"repeat_{repeat}"
            exit_code = int((result_dir / "exit_code.txt").read_text().strip())
            data = load_result(result_dir / "stdout.txt")
            repeat_data.append(data)
            measurements.append(flatten_measurement(
                case["split"], case["case_id"], repeat, data,
                result_dir / "stdout.txt", result_dir / "stderr.txt", exit_code))
            if exit_code != 0 or data["exact_bit_mismatches"] != 0:
                adapter_pass = False
        if repeat_data[0] != repeat_data[1]:
            determinism_pass = False
        parsed_by_case[case["case_id"]] = repeat_data[0]

    write_csv(arm_dir / "simulator_measurements.csv", measurements)
    canonical_rows = [row for row in measurements if row["repeat"] == 1]
    development = [row for row in canonical_rows if row["split"] == "development"]
    validation = [row for row in canonical_rows if row["split"] == "validation"]

    models = {}
    coefficients = []
    for kernel in ("ADD", "RELU", "GEMV"):
        models[kernel] = {}
        for phase in PHASES:
            model = fit_ols(development, kernel, phase)
            models[kernel][phase] = model
            coefficients.append(model)
    (arm_dir / "analytical_model_coefficients.json").write_text(
        json.dumps({
            "fit_split": "development_only",
            "locked_validation_used_for_fit": False,
            "models": coefficients,
        }, indent=2) + "\n"
    )

    comparison = []
    for row in canonical_rows:
        predictions = {
            phase: predict(models[row["kernel"]][phase], row) for phase in PHASES
        }
        predicted_total = sum(predictions.values())
        observed_total = row["native_total_cycles"]
        signed_error = predicted_total - observed_total
        relative_error = signed_error / observed_total
        record = {
            "split": row["split"],
            "case_id": row["case_id"],
            "kernel": row["kernel"],
            "analytical_initial_residency_cycles": predictions["initial_residency"],
            "simulator_initial_residency_cycles": row["initial_residency_cycles"],
            "initial_residency_error_cycles": predictions["initial_residency"] - row["initial_residency_cycles"],
            "analytical_kernel_execution_cycles": predictions["kernel_execution"],
            "simulator_kernel_execution_cycles": row["kernel_execution_cycles"],
            "kernel_execution_error_cycles": predictions["kernel_execution"] - row["kernel_execution_cycles"],
            "analytical_result_readback_cycles": predictions["result_readback"],
            "simulator_result_readback_cycles": row["result_readback_cycles"],
            "result_readback_error_cycles": predictions["result_readback"] - row["result_readback_cycles"],
            "analytical_total_cycles": predicted_total,
            "simulator_total_cycles": observed_total,
            "signed_error_cycles": signed_error,
            "absolute_error_cycles": abs(signed_error),
            "relative_error": relative_error,
            "absolute_relative_error": abs(relative_error),
            "fit_membership": "CALIBRATION" if row["split"] == "development" else "LOCKED_VALIDATION",
        }
        comparison.append(record)
    write_csv(arm_dir / "analytical_vs_simulator_validation.csv", comparison)

    validation_table = [row for row in comparison if row["split"] == "validation"]
    mape = sum(row["absolute_relative_error"] for row in validation_table) / len(validation_table)
    max_error = max(row["absolute_relative_error"] for row in validation_table)
    analytical_pass = mape <= 0.10 and max_error <= 0.20

    preliminary = []
    by_case = {row["case_id"]: row for row in validation}
    for case_id in ("conv_layer4_0_conv2", "relu_layer1_0_post_add",
                    "add_layer3_0_residual", "fc_classifier"):
        row = by_case[case_id]
        data = parsed_by_case[case_id]
        preliminary.append({
            "workload_id": case_id,
            "mapping_status": ("SUPPORTED_NATIVE_KERNEL_WITH_UNCOSTED_IM2COL" if case_id.startswith("conv_")
                               else "SUPPORTED_NATIVE_KERNEL_WITH_UNCOSTED_PARTIAL_SUM_REDUCTION" if data["kernel"] == "GEMV"
                               else "SUPPORTED_NATIVE_KERNEL"),
            "evidence_label": "SIMULATED_PIM_HBM2",
            "initial_residency_cycles": row["initial_residency_cycles"],
            "kernel_execution_cycles": row["kernel_execution_cycles"],
            "result_readback_cycles": row["result_readback_cycles"],
            "native_total_cycles": row["native_total_cycles"],
            "native_total_microseconds_at_tck_1ns": row["native_total_cycles"] / 1000.0,
            "exact_bit_mismatches": row["exact_bit_mismatches"],
            "outside_pim_uncosted_operations": row["outside_pim_uncosted_operations"],
            "end_to_end_latency_available": False if data["kernel"] == "GEMV" else True,
        })
    preliminary.append({
        "workload_id": "avgpool_global_control",
        "mapping_status": "UNSUPPORTED",
        "evidence_label": None,
        "initial_residency_cycles": None,
        "kernel_execution_cycles": None,
        "result_readback_cycles": None,
        "native_total_cycles": None,
        "native_total_microseconds_at_tck_1ns": None,
        "exact_bit_mismatches": None,
        "outside_pim_uncosted_operations": None,
        "end_to_end_latency_available": False,
    })
    write_csv(arm_dir / "preliminary_results.csv", preliminary)

    trace_summary = summarize_traces(run_dir, validation, arm_dir / "transaction_trace_summary.csv")
    quality = quality_accounting(arm_dir, arm_dir / "provenance/quality_run_accounting.json")

    expected_hashes = {
        "phase2_bundle": "8f0709f39267e79414cf9b7c9b2bde675b42531ffa45d90c0c71b730ca50f97b",
        "diagnostic_bundle": "6d64351133f952431d478162f644b0ddab6b5a3fb3fca40190d0fb5dd7fc8e96",
        "shape_source": "6a6a4bec2b90a791b92023382a5fbb45354a0c0e629e6c84164ee44b809fae2d",
        "quality_json": "942093102afd8a850bfd790f41adc6c6617486379518dd5ee00e4ae056102501",
        "database": "401a06b517765a9811a4f489ba9b7dde58a821dc40bb100d94ca27ff8660ca2b",
    }
    observed_hashes = {
        "phase2_bundle": (run_dir / "environment/phase2_bundle_sha256.txt").read_text().strip(),
        "diagnostic_bundle": (run_dir / "environment/diagnostic_bundle_sha256.txt").read_text().strip(),
        "shape_source": (run_dir / "environment/shape_source_sha256.txt").read_text().strip(),
        "quality_json": (run_dir / "environment/quality_json_sha256.txt").read_text().strip(),
        "database": (run_dir / "environment/database_sha256.txt").read_text().strip(),
    }
    provenance_pass = expected_hashes == observed_hashes
    end_to_end_mapping_pass = False
    representation_gate_pass = False
    gate_status = {
        "run_id": args.run_id,
        "status": "STOPPED_BEFORE_JOINT_OPTIMIZATION",
        "adapter_gate": {
            "pass": adapter_pass and determinism_pass,
            "all_exact_bit_mismatches_zero": adapter_pass,
            "two_repeat_cycle_and_output_determinism": determinism_pass,
        },
        "analytical_validation_gate": {
            "pass": analytical_pass,
            "locked_validation_case_count": len(validation_table),
            "mape": mape,
            "maximum_absolute_relative_error": max_error,
            "mape_limit": 0.10,
            "per_case_limit": 0.20,
            "locked_validation_used_for_fit": False,
        },
        "provenance_gate": {
            "pass": provenance_pass,
            "expected": expected_hashes,
            "observed": observed_hashes,
        },
        "end_to_end_mapping_gate": {
            "pass": end_to_end_mapping_pass,
            "reason": "Conv im2col/patch packing and GEMV host partial-sum reductions are counted but have no validated cycle cost; avgpool is unsupported."
        },
        "representation_gate": {
            "pass": representation_gate_pass,
            "validated_native_representations": ["FP16_dense"],
            "required_minimum": 2,
            "reason": "FP8/INT8/INT4/block-scaled/sparse formats have no validated native mapping, packing, or datapath."
        },
        "joint_optimizer": {
            "run": False,
            "reason": "End-to-end mapping and representation gates failed. Unsupported cases are not converted to zero-cost or zero-gap results."
        },
        "quality_record_accounting": {
            "directional_records_visible": quality["directional_record_count"],
            "unique_physical_run_ids": quality["unique_physical_run_id_count"],
            "statistical_samples_used": quality["statistical_sample_count_used_by_phase3"],
        },
        "trace_case_count": len(trace_summary),
    }
    # This is deliberately a narrow native-scope gate record.  The broader
    # Phase-3 specification is audited separately in gate_status.json and may
    # fail even when these originally tracked inputs match.
    (arm_dir / "gate_status_native_scope.json").write_text(json.dumps(gate_status, indent=2) + "\n")
    make_plots(arm_dir, validation, validation_table)

    print(json.dumps({
        "adapter_gate": gate_status["adapter_gate"],
        "analytical_validation_gate": gate_status["analytical_validation_gate"],
        "provenance_gate": gate_status["provenance_gate"],
        "end_to_end_mapping_gate": gate_status["end_to_end_mapping_gate"],
        "representation_gate": gate_status["representation_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
