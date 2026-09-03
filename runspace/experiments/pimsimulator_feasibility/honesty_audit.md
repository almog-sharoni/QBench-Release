# Honesty audit

## Disposition

The arm is stopped with status `STOPPED_REFERENCE_VALIDATION_FAILED`. It is not an approved PIM cost source for Stage 1.

## Audit checks

- [x] Work is isolated under `runspace/experiments/pimsimulator_feasibility/`.
- [x] Frozen L40S, H100, quantization-quality, and parametric bundles were not modified, overwritten, mixed, or reinterpreted.
- [x] Upstream URL, branch, commit, archive hash, environment, dependencies, commands, attempts, and local patches are recorded.
- [x] Locked upstream source remained clean; local simulator patches are an empty list.
- [x] Every attempted build and reference run preserved stdout, stderr, configuration copies, and exit codes.
- [x] The failed ADD performance test remains a failure; its threshold was not patched or waived.
- [x] Partial passing results are diagnostic only and cannot enter the optimizer.
- [x] All reported numeric observations are labeled `SIMULATED_PIM_HBM2`.
- [x] Nothing is labeled `MEASURED_HOST`, `MEASURED_PIM`, or `PROTOTYPE_MEASURED`.
- [x] No L40S/H100 bandwidth or compute throughput was transferred into this arm.
- [x] No LPDDR5X-PIM or physical-prototype validation claim is made.
- [x] INT4, FP8, block scaling, sparse execution, LPDDR5X-PIM, and generic PNM are marked `UNSUPPORTED` rather than assigned latency.
- [x] INT8 is not claimed as native: its parsed configuration token is insufficient because fallback arithmetic uses FP16 lane operators.
- [x] No low-precision traffic or cycle reduction was assumed.
- [x] No unsupported result was converted into zero cost or zero gap.
- [x] No workload shape was selected after seeing a joint gap; no joint gap was computed.
- [x] No post-outcome timing or mapping parameter tuning occurred.
- [x] No large joint-optimization sweep was launched.

## Statistical identity rule

Directional JSON rows with the same physical `run_id` are reconstructed views of one experiment. They must be joined/deduplicated by physical `run_id` for statistical counting and must never be treated as independent measurements, replicates, or samples. This arm did not ingest those JSON records.

## Consumption guard

`source_lock.json` sets both `downstream_cost_profile_authorized` and `joint_optimizer_integration_authorized` to `false`. `workload_shape_manifest.json` contains no shapes, and the analytical comparison CSV contains no numeric comparison. These are deliberate fail-closed states, not missing values to impute.
