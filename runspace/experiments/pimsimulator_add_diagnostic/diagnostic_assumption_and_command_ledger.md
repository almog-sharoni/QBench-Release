# Diagnostic assumption and command ledger

This ledger is append-only. Corrections are appended; prior entries are not removed or rewritten.

## 2026-08-23T16:20:00Z — arm creation and scope lock

- Created an isolated post-validation diagnostic arm at `runspace/experiments/pimsimulator_add_diagnostic/`.
- Frozen input arm: `runspace/experiments/pimsimulator_feasibility/`.
- No file in the frozen input arm may be modified, including its locked upstream clone, raw outputs, reports, manifests, or container image.
- The diagnostic will use the same upstream commit `3703d1f19c8f027360cc33a3243eb271e3bb6898` and the same pinned container image by read-only reference.
- The strict upstream assertion will not be patched, relaxed, or reinterpreted as a functionality assertion.
- A fresh source checkout and fresh build directory will be used for the clean reproduction.
- No workload adapter or joint-optimization result will be created.

## 2026-08-23T16:21:27Z — clean diagnostic run

- Created fresh run `add_diagnostic_20260823T163000Z` with `exact_reproduction.sh`.
- Verified fresh clone commit `3703d1f19c8f027360cc33a3243eb271e3bb6898`, source archive SHA-256 `803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392`, and a clean source status before building.
- Verified pinned container SHA-256 `5e9455a5c0cfdf9e68c818473f66498bc1b7e911850f85e5b30a40091ec12db4`.
- Built an exported source tree in a new run directory. The upstream source checkout was not built in place or edited.
- Ran the functional ADD test once, the performance ADD test five times with one unchanged binary, and the complete upstream suite. Preserved commands, stdout, stderr, configurations, hashes, and exit codes.

## 2026-08-23T16:24:00Z — exact-output diagnostic

- The upstream functional comparator is tolerant, so added an external diagnostic harness at `probes/exact_add_compare.cpp`; this is not an upstream patch.
- Linked the harness against the unchanged clean-build simulator library and compared all 1,048,576 FP16 output bit patterns with both the locked NumPy reference and an independently recomputed FP16 addition.
- Preserved two failed compile attempts caused solely by inaccessible host paths inside the container. The third attempt used an explicit bind and succeeded. No source or timing behavior was changed to make it succeed.
- The harness includes preload and result-readback traffic and reports 9,626 cycles. That number is not comparable to, and must not replace, the upstream performance benchmark's 3,349-cycle PIM path.

## 2026-08-23T16:25:06Z — transaction trace probe

- Copied the clean exported build into `trace_probe/source` and changed only three output controls in `system_hbm_64ch.ini`: `SHOW_SIM_OUTPUT=false` to `true`, `PRINT_CHAN_STAT=true` to `false`, and `PRINT_MEM_TRACE=true` to `false`.
- The simulator binary remained byte-identical: SHA-256 `e348be1c030166f3ed6eee7f2632f996f21472c4aecdb2620797b4920944ae9b`.
- Re-ran only `PIMBenchFixture.add`. The observed cycles remained 6,651 non-PIM and 3,349 PIM, and the unmodified assertion still exited 1.
- The trace probe is diagnostic logging, not a new measurement or independent statistical sample.

## 2026-08-23T16:40:00Z — provenance inspection

- Inspected local blame, file history, threshold pickaxe history, branch/tag state, documentation, and the commit that changed `parkOut()` to all-bank operation.
- Captured the public GitHub branch, tag, issue, issue-comment, pull-request, and pull-request-diff API responses under `provenance/github_api/` without editing them.
- Did not switch to an older commit to seek a passing result and did not infer an alternate timing configuration from unrelated reports.

## 2026-08-23T17:00:00Z — classification and stop

- Selected exactly one classification: `UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`.
- Preserved the original Phase-2 validation failure and the observed 3,349-cycle value.
- Recommended only a future, separately reviewed and preregistered validation stage under the explicit label `VALIDATED_WITH_UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`.
- Stopped without creating a workload adapter, analytical cost integration, or joint-optimization sweep.

## 2026-08-23T16:36:41Z — timestamp correction

- Correction: the preceding `16:40:00Z` and `17:00:00Z` headings were drafted as ordering labels but are later than the actual completion time. The provenance inspection and classification deliverables were completed by `2026-08-23T16:36:41Z`. The earlier entries are retained because this ledger is append-only.

## 2026-08-23T16:36:41Z — frozen-arm verification

- Rechecked all 2,506 regular files from the frozen Phase-2 feasibility arm against `frozen_phase2_sha256sums.txt` with `sha256sum --quiet -c`.
- Result: PASS. Manifest SHA-256: `8f0709f39267e79414cf9b7c9b2bde675b42531ffa45d90c0c71b730ca50f97b`.
