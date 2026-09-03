# PIMSimulator HBM2 feasibility arm

Status: **stopped — unmodified reference validation failed**.

The locked simulator built successfully, but `PIMBenchFixture.add` failed its upstream strictly-greater-than-2× assertion at 1.985965967×. Under the preregistered stop rule, this arm produced no Stage 1 cost profile, adapter, analytical calibration, or joint-optimization sweep.

Primary review files:

- `source_lock.json`
- `simulator_assumption_and_patch_ledger.md`
- `simulator_validation_report.md`
- `workload_shape_manifest.json`
- `analytical_vs_simulator_validation.csv`
- `supported_configuration_matrix.csv`
- `honesty_audit.md`
- `limitations_and_unsupported_claims.md`
- `reproduce_reference_validation.sh`
- `raw_output_sha256sums.txt`
- `artifacts/raw/`

The arm is isolated from frozen L40S, H100, quantization-quality, and parametric result bundles. Directional JSON views sharing one physical `run_id` remain one experiment for statistical purposes and were not ingested here.
