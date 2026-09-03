# PIMSimulator ADD post-validation diagnostic arm

Status: `COMPLETE — STOPPED AFTER CLASSIFICATION`.

This append-only arm investigates the preserved `PIMBenchFixture.add` failure from the frozen Phase-2 feasibility arm. It must not modify that arm or patch the upstream `speedup > 2.0` assertion. It stops after classification and recommendation; workload mapping and joint optimization are out of scope.

The four permitted final classifications are:

- `FUNCTIONAL_FAILURE`
- `NONDETERMINISTIC_SIMULATION`
- `CONFIGURATION_MISMATCH`
- `UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`

Final classification: `UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`.

The diagnostic preserves the upstream failure. It does not change the `> 2.0` assertion or substitute the threshold-implied cycle count for the observed 3,349 cycles. The proposed label for any separately reviewed future validation stage is `VALIDATED_WITH_UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`.

Primary deliverables:

- `diagnostic_report.md`
- `failing_assertion_source.txt`
- `repeated_run_results.csv`
- `clean_rebuild_report.md`
- `threshold_provenance.md`
- `diagnostic_classification.json`
- `updated_honesty_audit.md`
- `exact_reproduction.sh`

Supporting inventories include `command_inventory.md`, `source_hashes.txt`, the append-only `diagnostic_assumption_and_command_ledger.md`, and `frozen_phase2_verification.txt`.
