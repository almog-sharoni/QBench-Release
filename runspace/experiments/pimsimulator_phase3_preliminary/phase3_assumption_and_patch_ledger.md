# Phase-3 assumption and patch ledger

This ledger is append-only. Corrections are appended; prior entries are never removed or rewritten.

## 2026-08-23T16:48:02Z — authorization and input lock

- Accepted user authorization label: `VALIDATED_WITH_UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`.
- Created isolated arm `runspace/experiments/pimsimulator_phase3_preliminary/`.
- Locked the feasibility bundle at 2,506 files / 644,605,039 bytes / bundle SHA-256 `8f0709f39267e79414cf9b7c9b2bde675b42531ffa45d90c0c71b730ca50f97b`.
- Locked the diagnostic bundle at 539 files / 608,717,419 bytes / bundle SHA-256 `6d64351133f952431d478162f644b0ddab6b5a3fb3fca40190d0fb5dd7fc8e96`.
- The hash definition is recorded in `phase3_manifest.json` and includes every regular file, including nested `.git` files.
- Preserved the upstream ADD failure as exit 1, 6,651 non-PIM cycles, 3,349 PIM cycles, strict `>2.0`, and 24 final all-bank park-out drain cycles.
- No upstream patch is allowed. All Phase-3 adapter code will remain outside the exported upstream tree.

## 2026-08-23T16:48:02Z — workload and analysis preregistration

- Fixed five ResNet-50 representatives before any Phase-3 simulator result: layer4.0.conv2, the first-stage post-residual ReLU, layer3.0 residual ADD, fc, and avgpool as an unsupported control.
- Fixed the development/calibration and locked-validation cases in `preregistration.json`.
- Fixed the analytical features, pass thresholds, trace policy, and downstream stop gates.
- Low-precision records may be inspected for provenance and quality only. They do not imply PIM hardware support or reduced PIM traffic/cycles.
- Directional JSON duplicates sharing a physical `run_id` will remain visible in source provenance but will be counted once if any run-level summary is made.
- No joint-gap result has been observed. No simulator or analytical parameter may be tuned in response to a future gap without a new post-result preregistration.

## 2026-08-23T16:48:02Z — pre-execution GEMV mapping correction

- Before any Phase-3 simulator execution, replaced multi-batch GEMV wording with sequential batch-1 GEMV invocations that retain weights and read each result before reusing the result location.
- Reason: upstream reference validation covers batch 1, while its functionality fixture does not validate all batch outputs. The more conservative sequential mapping stays within the validated invocation pattern and makes per-vector mode/CRF/readback overhead visible.
- Updated development and validation field name `batch` to `vectors`; no shape selection, pass threshold, or result-driven parameter changed.

## 2026-08-23T16:52:00Z — corrected preflight path incident

- A preflight shell command created five nested empty directories under the diagnostic run's exported build snapshot because its relative `mkdir` target was resolved from that working directory.
- Inspection confirmed the new path contained directories only and no regular files. Removed exactly those empty directories with explicit `rmdir` calls; no existing file was removed or edited.
- Immediately recomputed the full 539-file diagnostic bundle digest. It remained exactly `6d64351133f952431d478162f644b0ddab6b5a3fb3fca40190d0fb5dd7fc8e96`.
- All subsequent Phase-3 output paths are absolute or rooted under `/phase3` inside the explicit Phase-3 bind.

## 2026-08-23T16:54:00Z — adapter compile preflight

- Compiled `adapter/phase3_runner.cpp` against the unchanged locked simulator static library with the upstream compiler flags plus the adapter source.
- Adapter source SHA-256: `9ffe2119d022ad73cdc5d2f7f5c9e2f8e6ebc599916f715d91bf4c0673b6997a`.
- Successful preflight binary SHA-256: `b4bf0e6436174e59f432306647d8bd7bf19a6a562f2ad40224e4897963fc22fe`.
- Preserved the sandbox/runtime and missing-output-directory failures in `raw/preflight/compile_notes.txt`; neither failure modified simulator source.

## 2026-08-23T17:15:03Z — Phase-3 capture and locked analysis

- Executed formal run `phase3_preliminary_20260823T165500Z` from a clean source export. The simulator source was not patched. The adapter remained external to the export.
- Reproduced all five `PIMKernelFixture.*` functional tests. Reproduced the upstream ADD performance deviation unchanged: exit 1, strict `> 2.0`, 6,651 non-PIM cycles, and 3,349 PIM cycles. The accepted diagnostic bundle retains the identified 24-cycle final all-bank park-out drain.
- All 30 adapter executions (15 cases, two determinism repeats) exited 0, had zero bit mismatches, and reproduced identical cycles and outputs. Repeats are paired determinism checks, not independent samples.
- Preserved gzip-compressed transaction traces for all four locked validation cases. Trace logging used a copied source/build tree and a recorded logging-only configuration diff; it did not modify the frozen source repository.
- Fit the preregistered ordinary-least-squares phase models using only the 11 development cases. No locked case was used for fitting and no feature or threshold changed after results were observed.
- Locked validation passed: four-case total-cycle MAPE 2.5573%; maximum absolute relative error 4.9226%.
- Provenance passed: the frozen feasibility and diagnostic bundle hashes and all tracked shape/quality/database hashes matched their preregistered values.
- Stopped before joint optimization. End-to-end mapping failed because im2col/patch packing and GEMV host partial-sum reductions are counted but uncosted, and avgpool is unsupported. Representation support failed because dense FP16 is the only validated native representation.
- No positive, zero, or negative joint gap was computed. The result is `NOT_EVALUATED`, not zero.

## 2026-08-23T17:27:06Z — full-specification completion audit and superseding stop

- Read the complete Phase-3 continuation specification supplied after the initial narrow-scope bundle was assembled.
- Found that the initial preregistration did not include all required residency modes, parameter sensitivity ranges, quality constraints, fixed-point convergence conditions, tie-breaking rules, result classifications, or required figures/reports. Recorded these as missing rather than retroactively preregistering them.
- Searched the current workspace for required frozen L40S, H100, parametric, Stage-1-plan, checkpoint, and locked 48,000-image split manifests. They were not located; no hash was invented.
- Did not enumerate, open, decode, or evaluate the locked final-evaluation dataset. Because its manifest was unavailable, even the permitted manifest-only hash guard could not be performed.
- Corrected the full-input provenance decision from the earlier limited-scope pass to fail. The original narrow completion manifest SHA-256 is `049687e9a914d4aa652d6d80d7b18811aa89c85e94039b0601c4b644d04b4082`; it remains present and is explicitly superseded by the full-spec completion manifest.
- Preserved the valid native simulator capture and analytical acceptance result without refitting, rerunning, or tuning it.
- Generated descriptive and presentation artifacts only. No simulator parameter, workload selection, quality budget, optimizer parameter, joint result, or ablation result was added or tuned.
- Marked host provenance unresolved where the available JSON could not prove `MEASURED_HOST_L40S` versus `EMULATED_HOST_L40S`.
- Stopped with `STOPPED_FULL_SPEC_PROVENANCE_AND_PREREGISTRATION_FAILED`. No regime claim is made.
