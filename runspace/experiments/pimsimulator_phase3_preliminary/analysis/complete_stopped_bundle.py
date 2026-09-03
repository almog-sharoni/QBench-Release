#!/usr/bin/env python3
"""Produce presentation artifacts for the stopped Phase-3 arm.

This script does not execute PIMSimulator, inspect an ImageNet split, fit a new
model, or run an optimizer.  It only renders frozen Phase-3 CSV/JSON evidence
and emits explicit unavailable/unsupported records required by the Phase-3
presentation specification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def size_group(case_id: str, measurements: dict[str, dict[str, str]]) -> str:
    row = measurements[case_id]
    byte_fields = [
        "traffic_padded_input_bytes",
        "traffic_padded_output_readback_bytes",
        "traffic_padded_weight_residency_bytes",
        "traffic_padded_input_upload_bytes",
        "traffic_partial_sum_readback_bytes",
    ]
    total = sum(int(float(row.get(field) or 0)) for field in byte_fields)
    if total <= 1024 * 1024:
        return "LE_1_MiB"
    if total <= 32 * 1024 * 1024:
        return "GT_1_TO_32_MiB"
    return "GT_32_MiB"


def analytical_outputs(arm: Path) -> None:
    validation = read_csv(arm / "analytical_vs_simulator_validation.csv")
    measurements_all = read_csv(arm / "simulator_measurements.csv")
    measurements = {
        row["case_id"]: row
        for row in measurements_all
        if row["repeat"] == "1"
    }

    output_rows: list[dict[str, object]] = []
    for row in validation:
        enriched = dict(row)
        enriched["tensor_size_group"] = size_group(row["case_id"], measurements)
        enriched["prediction_label"] = "CALIBRATED_HYBRID_MODEL"
        enriched["observation_label"] = "SIMULATED_PIM_HBM2"
        output_rows.append(enriched)
    fields = list(validation[0]) + [
        "tensor_size_group",
        "prediction_label",
        "observation_label",
    ]
    write_csv(arm / "analytical_vs_simulated_validation.csv", fields, output_rows)

    locked = [row for row in output_rows if row["split"] == "validation"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in locked:
        error = float(row["absolute_relative_error"])
        grouped[("overall", "all")].append(error)
        grouped[("operation", str(row["kernel"]))].append(error)
        grouped[("tensor_size", str(row["tensor_size_group"]))].append(error)
    stats: list[dict[str, object]] = []
    for (group_type, group), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        stats.append(
            {
                "evaluation_split": "locked_validation",
                "group_type": group_type,
                "group": group,
                "case_count": len(values),
                "mean_absolute_relative_error": float(np.mean(arr)),
                "median_absolute_relative_error": float(np.median(arr)),
                "worst_absolute_relative_error": float(np.max(arr)),
                "descriptive_grouping_preregistered": group_type != "tensor_size",
                "acceptance_test_role": "OVERALL_ONLY" if group_type == "overall" else "DESCRIPTIVE_ONLY",
            }
        )
    write_csv(
        arm / "analytical_validation_statistics.csv",
        list(stats[0]),
        stats,
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    colors = {"development": "#4C78A8", "validation": "#F58518"}
    for split in ("development", "validation"):
        rows = [row for row in output_rows if row["split"] == split]
        simulated = np.asarray([float(row["simulator_total_cycles"]) for row in rows])
        predicted = np.asarray([float(row["analytical_total_cycles"]) for row in rows])
        ax.scatter(simulated, predicted, label=split, color=colors[split], s=58, alpha=0.9)
    all_values = [
        float(row[key])
        for row in output_rows
        for key in ("simulator_total_cycles", "analytical_total_cycles")
    ]
    lower, upper = min(all_values) * 0.8, max(all_values) * 1.25
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("PIMSimulator cycles (SIMULATED_PIM_HBM2)")
    ax.set_ylabel("Predicted cycles (CALIBRATED_HYBRID_MODEL)")
    ax.set_title("Analytical prediction versus PIMSimulator\n(all development and locked cases; no outliers removed)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(arm / "analytical_vs_simulated_latency.pdf")
    plt.close(fig)


def physical_run_outputs(arm: Path, project: Path) -> None:
    source = project / "runspace/experiments/find_optimal_hybrid_quant/results/latest_db_results.json"
    records = json.loads(source.read_text())
    model = next(record for record in records if record["model_name"] == "resnet50")
    rows: list[dict[str, object]] = []
    run_counts = Counter(str(item["run_id"]) for item in model["hybrid_results"])
    for logical_index, item in enumerate(model["hybrid_results"]):
        run_id = str(item["run_id"])
        rows.append(
            {
                "model_name": "resnet50",
                "logical_record_index": logical_index,
                "physical_run_id": run_id,
                "physical_cluster_size": run_counts[run_id],
                "selection_direction": item.get("selection_direction"),
                "reconstructed_duplicate": str(bool(item.get("reconstructed_duplicate"))).lower(),
                "weight_type": item.get("weight_type"),
                "weight_mode": item.get("weight_mode"),
                "input_type": item.get("input_type"),
                "input_mode": item.get("input_mode"),
                "bit_width": item.get("bit_width"),
                "experiment_type": item.get("experiment_type"),
                "acc1": item.get("acc1"),
                "acc5": item.get("acc5"),
                "mse": item.get("mse"),
                "l1": item.get("l1"),
                "host_provenance_label": "UNRESOLVED_HOST_PROVENANCE",
                "independent_measurement_in_phase3": "false",
                "independent_statistical_sample_in_phase3": "false",
                "pim_cost_candidate": "false",
                "pim_format_status": "UNSUPPORTED_PIM_FORMAT" if item.get("bit_width") != 32 else "NOT_MAPPED_TO_PIM",
            }
        )
    write_csv(arm / "physical_run_provenance.csv", list(rows[0]), rows)

    formats: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in model["hybrid_results"]:
        formats[("weight", str(item.get("weight_type")))].add(int(item.get("bit_width") or 0))
        formats[("input", str(item.get("input_type")))].add(int(item.get("bit_width") or 0))
    representation_rows: list[dict[str, object]] = []
    for (domain, representation), observed_widths in sorted(formats.items()):
        is_fp32 = representation == "fp32"
        representation_rows.append(
            {
                "tensor_domain": domain,
                "representation": representation,
                "observed_result_bit_width_groups": ";".join(str(value) for value in sorted(observed_widths)),
                "host_quality_record_status": "PRESENT_PROVENANCE_UNRESOLVED",
                "pim_mapping_status": "NOT_MAPPED_TO_PIM" if is_fp32 else "UNSUPPORTED_PIM_FORMAT",
                "pim_transaction_packing_validated": "false",
                "pim_datapath_validated": "false",
                "pim_scale_or_zero_point_metadata_bytes": "",
                "pim_block_scaling_metadata_bytes": "",
                "pim_sparse_index_metadata_bytes": "",
                "pim_latency": "",
                "unavailable_values_encoded_as_zero": "false",
            }
        )
    representation_rows.extend(
        [
            {
                "tensor_domain": domain,
                "representation": "fp16_dense",
                "observed_result_bit_width_groups": "",
                "host_quality_record_status": "NOT_PRESENT_IN_DIRECTIONAL_QUALITY_JSON",
                "pim_mapping_status": "VALIDATED_NATIVE_PIM_REPRESENTATION",
                "pim_transaction_packing_validated": "true",
                "pim_datapath_validated": "true",
                "pim_scale_or_zero_point_metadata_bytes": 0,
                "pim_block_scaling_metadata_bytes": 0,
                "pim_sparse_index_metadata_bytes": 0,
                "pim_latency": "See simulator_measurements.csv",
                "unavailable_values_encoded_as_zero": "false",
            }
            for domain in ("input", "weight", "output", "partial_sum")
        ]
    )
    write_csv(
        arm / "representation_validity_matrix.csv",
        list(representation_rows[0]),
        representation_rows,
    )


def stopped_tables(arm: Path) -> None:
    preliminary = read_csv(arm / "preliminary_results.csv")
    rows: list[dict[str, object]] = []
    for row in preliminary:
        rows.append(
            {
                **row,
                "result_classification": "INCONCLUSIVE_STOPPED_ARM",
                "joint_optimum": "",
                "placement_first_fixed_point": "",
                "representation_first_fixed_point": "",
                "best_sequential": "",
                "absolute_joint_gap": "",
                "relative_joint_gap": "",
                "joint_gap_status": "NOT_EVALUATED",
                "reason": "End-to-end, representation, and full-input provenance gates did not pass.",
            }
        )
    write_csv(arm / "preliminary_results_table.csv", list(rows[0]), rows)

    mechanisms = [
        ("initial_and_persistent_residency", "PIMSimulator initial residency; Stage-1 persistence policy not validated", "SIMULATED_PIM_HBM2 / TRACE_DERIVED"),
        ("asymmetric_conversion_and_packing", "No frozen host conversion/packing bundle was located", "UNAVAILABLE_PROVENANCE"),
        ("placement_dependent_metadata_traffic", "No supported low-precision PIM mapping", "UNSUPPORTED_PIM_FORMAT"),
        ("format_feasibility", "Only dense FP16 is simulator-validated", "SIMULATED_PIM_HBM2"),
        ("pim_compute_ceilings", "Native cycle observations exist; no alternate validated representation", "SIMULATED_PIM_HBM2"),
        ("boundary_synchronization", "Included in aggregate adapter phases but not separately cycle-attributed", "SIMULATED_PIM_HBM2_NOT_SEPARABLE"),
        ("pim_mode_transitions", "Commands preserved in traces; cycles not separately attributed", "SIMULATED_PIM_HBM2_NOT_SEPARABLE"),
        ("all_bank_parkout_drain", "24-cycle diagnostic source retained; per-workload contribution not separately attributed", "SIMULATED_PIM_HBM2_DIAGNOSTIC"),
        ("capacity_constraints", "No validated Stage-1 placement/capacity candidate space", "UNAVAILABLE_PROVENANCE"),
    ]
    ablations = [
        {
            "mechanism": mechanism,
            "physical_source": source,
            "provenance": provenance,
            "ablation_status": "DEFERRED_STOPPED_BEFORE_OPTIMIZATION",
            "unrelated_parameters_changed": "false",
            "joint_gap_before": "",
            "joint_gap_after": "",
            "joint_gap_change": "NOT_EVALUATED",
            "causal_claim": "NONE",
        }
        for mechanism, source, provenance in mechanisms
    ]
    write_csv(arm / "mechanism_ablation.csv", list(ablations[0]), ablations)

    claims = [
        ("host_fp32_quality", "Host FP32 quality exists in directional JSON", "", "UNRESOLVED_HOST_PROVENANCE", "NOT_ADMISSIBLE", "Frozen L40S bundle/hash and device evidence not located"),
        ("host_quantized_quality", "Quantized host quality exists in directional JSON", "", "UNRESOLVED_HOST_PROVENANCE", "NOT_ADMISSIBLE", "Cannot prove EMULATED_HOST_L40S label from JSON alone"),
        ("tensor_shapes", "Representative ResNet-50 shapes", "workload_shape_manifest.json", "TRACE_DERIVED", "ADMISSIBLE_WITH_LIMITATION", "Checkpoint identity is not fully frozen"),
        ("pim_native_cycles", "ADD/ReLU/GEMV native phase cycles", "simulator_measurements.csv", "SIMULATED_PIM_HBM2", "ADMISSIBLE", "Exact locked commit/configuration only"),
        ("analytical_prediction", "Development-fit native cycle model", "analytical_vs_simulated_validation.csv", "CALIBRATED_HYBRID_MODEL", "ADMISSIBLE_NATIVE_SCOPE", "Not end-to-end"),
        ("quantization_metadata", "Low-precision scale/zero-point/block metadata", "", "ANALYTICAL_METADATA", "UNAVAILABLE", "No frozen format/packing mapping"),
        ("host_to_pim_boundary", "Host-to-PIM transfer latency", "", "ASSUMED_PARAMETRIC", "UNAVAILABLE", "No frozen parametric bundle/hash located"),
        ("pim_to_host_boundary", "PIM-to-host transfer latency", "", "ASSUMED_PARAMETRIC", "UNAVAILABLE", "No frozen parametric bundle/hash located"),
        ("low_precision_pim", "FP8/INT8/INT4/block/sparse PIM cost", "supported_configuration_matrix.csv", "UNSUPPORTED_PIM_FORMAT", "UNSUPPORTED", "No validated datapath/ISA/packing"),
        ("joint_gap", "Joint over converged sequential gap", "joint_gap_regime_map.pdf", "NOT_EVALUATED", "NO_CLAIM", "Prerequisite gates failed"),
    ]
    ledger_rows = [
        {
            "claim_id": claim_id,
            "claim": claim,
            "authoritative_artifact": artifact,
            "provenance_label": provenance,
            "admissibility": admissibility,
            "limitation_or_reason": reason,
        }
        for claim_id, claim, artifact, provenance, admissibility, reason in claims
    ]
    write_csv(arm / "provenance_and_claim_ledger.csv", list(ledger_rows[0]), ledger_rows)


def presentation_pdfs(arm: Path) -> None:
    results = read_csv(arm / "preliminary_results.csv")
    supported = [row for row in results if row["native_total_cycles"]]
    labels = [row["workload_id"].replace("_", "\n") for row in supported]
    initial = np.asarray([float(row["initial_residency_cycles"]) for row in supported])
    execute = np.asarray([float(row["kernel_execution_cycles"]) for row in supported])
    readback = np.asarray([float(row["result_readback_cycles"]) for row in supported])
    x = np.arange(len(supported))

    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2.2, 1.4]})
    ax.bar(x, initial, label="Initial residency", color="#4C78A8")
    ax.bar(x, execute, bottom=initial, label="Aggregate kernel execution", color="#F58518")
    ax.bar(x, readback, bottom=initial + execute, label="Aggregate result readback", color="#54A24B")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Simulator cycles (log scale)")
    ax.set_title("Available native FP16 HBM2-PIM components (not full host–PIM latency)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    component_rows = [
        ["Host compute", "UNAVAILABLE", "frozen L40S bundle not located"],
        ["Host memory traffic", "UNAVAILABLE", "frozen L40S/H100 bundle not located"],
        ["Conversion / packing", "UNAVAILABLE", "no validated host boundary cost"],
        ["Host→PIM traffic", "COUNTED BYTES; LATENCY UNAVAILABLE", "adapter / no frozen parametric source"],
        ["PIM mode + CRF", "INCLUDED; NOT SEPARABLE", "SIMULATED_PIM_HBM2 trace"],
        ["PIM memory / compute", "INCLUDED; NOT SEPARABLE", "SIMULATED_PIM_HBM2 trace"],
        ["Mode exit + parkOut", "INCLUDED; NOT SEPARABLE", "24-cycle diagnostic retained"],
        ["PIM→host + synchronization", "AGGREGATE READBACK ONLY", "SIMULATED_PIM_HBM2"],
    ]
    table_ax.axis("off")
    table = table_ax.table(
        cellText=component_rows,
        colLabels=["Requested component", "Status", "Provenance"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.25, 0.30, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    fig.suptitle("Host–PIM cost breakdown: explicit availability boundary", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(arm / "host_pim_cost_breakdown.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.08, 0.25), 0.84, 0.55, color="#E0E0E0", ec="#555555", lw=2))
    ax.text(0.5, 0.63, "INCONCLUSIVE — NOT EVALUATED", ha="center", va="center", fontsize=19, fontweight="bold")
    ax.text(
        0.5,
        0.46,
        "No joint or converged sequential optimizer ran.\n"
        "End-to-end mapping, representation support, and full provenance gates failed.",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.text(0.5, 0.32, "Unavailable and unsupported cells are not zero-gap cells.", ha="center", fontsize=11, color="#B22222")
    ax.set_title("Joint-gap regime map — stopped arm (no regime claim)", fontsize=15, pad=18)
    fig.tight_layout()
    fig.savefig(arm / "joint_gap_regime_map.pdf")
    plt.close(fig)

    fig = plt.figure(figsize=(13.333, 7.5))
    title_ax = fig.add_axes([0.04, 0.07, 0.48, 0.86])
    title_ax.axis("off")
    title_ax.text(0, 0.98, "Samsung HBM2-PIM Phase 3", fontsize=23, fontweight="bold", va="top")
    title_ax.text(0, 0.88, "Native feasibility passes; Stage-1\nregime claim remains inconclusive", fontsize=14, color="#444444", va="top")
    bullets = [
        "30/30 mapped executions: zero bit mismatches\nand deterministic cycles",
        "Locked validation: 2.56% MAPE;\n4.92% worst case",
        "Native FP16 only; low-precision PIM formats\nremain unsupported",
        "Im2col, host reductions, boundary costs, and\navgpool are not validated end-to-end",
        "Required host/parametric/plan/checkpoint/split\nprovenance was not located",
        "Joint and fixed-point sequential optimization:\nNOT EVALUATED (not zero gap)",
    ]
    y = 0.70
    for bullet in bullets:
        title_ax.text(0.02, y, "• " + bullet, fontsize=11.2, va="top")
        y -= 0.105
    title_ax.text(
        0,
        0.015,
        "Conclusion: simulator coverage and provenance are\n"
        "insufficient for a coupled-versus-separable claim.",
        fontsize=11.5,
        fontweight="bold",
        color="#8B0000",
    )

    cost_ax = fig.add_axes([0.59, 0.56, 0.37, 0.36])
    cost_ax.bar(x, initial, label="residency", color="#4C78A8")
    cost_ax.bar(x, execute, bottom=initial, label="execute", color="#F58518")
    cost_ax.bar(x, readback, bottom=initial + execute, label="readback", color="#54A24B")
    cost_ax.set_yscale("log")
    cost_ax.set_xticks(x, ["ADD", "ReLU", "FC", "Conv\nGEMV"])
    cost_ax.set_ylabel("cycles (log)")
    cost_ax.set_title("SIMULATED_PIM_HBM2 native portions")
    cost_ax.legend(fontsize=8, ncol=3)
    cost_ax.grid(True, axis="y", alpha=0.2)

    validation = [row for row in read_csv(arm / "analytical_vs_simulator_validation.csv") if row["split"] == "validation"]
    error_ax = fig.add_axes([0.59, 0.10, 0.37, 0.34])
    errors = [100 * float(row["absolute_relative_error"]) for row in validation]
    error_ax.bar(range(len(validation)), errors, color="#59A14F")
    error_ax.axhline(20, color="#E15759", linestyle="--", label="20% per-case gate")
    error_ax.set_xticks(range(len(validation)), ["ADD", "ReLU", "FC", "Conv"])
    error_ax.set_ylabel("absolute relative error (%)")
    error_ax.set_ylim(0, 21)
    error_ax.set_title("Locked analytical validation")
    error_ax.legend(fontsize=8)
    error_ax.grid(True, axis="y", alpha=0.2)

    fig.savefig(arm / "phase3_preliminary_defense_slide.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-dir", type=Path, required=True)
    args = parser.parse_args()
    arm = args.arm_dir.resolve()
    project = arm.parents[2]
    analytical_outputs(arm)
    physical_run_outputs(arm, project)
    stopped_tables(arm)
    presentation_pdfs(arm)
    print(json.dumps({"status": "STOPPED_BUNDLE_ARTIFACTS_RENDERED", "arm": str(arm)}, indent=2))


if __name__ == "__main__":
    main()
