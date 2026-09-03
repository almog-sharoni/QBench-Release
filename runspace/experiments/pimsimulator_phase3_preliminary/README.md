# PIMSimulator Phase-3 workload integration and preliminary results

Status: `STOPPED_FULL_SPEC_PROVENANCE_AND_PREREGISTRATION_FAILED`.

This is a new isolated arm authorized under `VALIDATED_WITH_UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`. It may use Samsung SAIT PIMSimulator commit `3703d1f19c8f027360cc33a3243eb271e3bb6898` only for the locked HBM2 FP16 configuration and only within the validity boundary in `preregistration.json`.

The frozen feasibility and ADD-diagnostic bundles are read-only inputs. Their prior failure remains visible: strict `speedup > 2.0`, exit code 1, 6,651 non-PIM cycles, 3,349 PIM cycles, and a 24-cycle final all-bank `parkOut` drain.

The native correctness/determinism and narrow analytical-validation gates passed. The locked four-case validation has 2.56% total-cycle MAPE and 4.92% maximum absolute relative error. The later full-spec audit found that the initial provenance check covered only locally identified inputs: required frozen L40S, H100, parametric, Stage-1-plan, checkpoint, and locked-split manifests were not located. Several required Phase-3A commitments were also absent before results. The arm therefore stops on full provenance/preregistration as well as incomplete end-to-end mapping and single-representation support. Unsupported formats or operations receive `UNSUPPORTED`, never zero latency or a zero joint gap.

Start with `preliminary_results_summary.md`. `phase3_full_spec_completion_manifest.json` is the authoritative completion decision; the earlier `phase3_completion_manifest.json` records the narrower audit scope and is superseded. `exact_reproduction.sh` rebuilds and captures the simulator evidence without modifying the frozen bundles or the upstream simulator source.
