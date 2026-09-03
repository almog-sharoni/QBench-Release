# PIMSimulator assumption and patch ledger

This ledger is append-only. Corrections are added as new dated entries; prior entries are not rewritten or deleted. No entry authorizes mixing this arm with the frozen L40S, H100, quantization-quality, or parametric bundles.

## 2026-08-23T15:55:36Z — source freeze

- Cloned `https://github.com/SAITPublic/PIMSimulator.git` with `--no-tags` into the isolated `upstream/PIMSimulator` directory.
- The upstream default branch resolved to `dev` at commit `3703d1f19c8f027360cc33a3243eb271e3bb6898`.
- The working tree was clean at lock time.
- The canonical source digest is SHA-256 `803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392`, computed over `git archive --format=tar HEAD`.
- Local simulator patches: **none**.
- Assumption A001: Ubuntu 24.04 packages `scons=4.5.2+dfsg-1`, `libgtest-dev=1.14.0-1`, and `googletest=1.14.0-1` are build dependencies, not simulator modifications. They will be extracted under this arm rather than installed system-wide.
- Assumption A002: a “clean environment” means a clean locked upstream tree plus isolated, versioned dependency and build directories with captured commands and environment; it does not imply that compiler/kernel performance represents PIM hardware.
- Gate G001: if any required unmodified bandwidth, GEMV, ADD, MUL, or ReLU reference test fails, this arm stops and no simulator-derived cost profile is produced.
- Data rule D001: reconstructed directional JSON records sharing a physical `run_id` are views of one experiment and are never independent measurements or statistical samples.

## 2026-08-23T15:57:39Z — preserved failed GCC 13 build

- Attempt `reference_validation_20260823T155536Z` used GCC 13.3.0 and failed during compilation with exit code 2.
- Raw failure logs are preserved under `artifacts/raw/reference_validation_20260823T155536Z/`.
- Failures were missing direct standard-library includes for `uint32_t` and `std::ostream_iterator` in the upstream sources. No source file was patched.
- This attempt did not reach test discovery or execute any reference test and therefore is not validation evidence.
- Assumption A003: because the source dates from 2021 and does not document a compiler version, a second clean, unmodified-source attempt with an isolated GCC 9.5.0 toolchain is permitted as a compiler-compatibility check. This decision was made before any simulator test or Stage 1 joint-gap result was observed.
- Local simulator patches remain **none**.

## 2026-08-23T16:13:00Z — correction to raw-digest entry placement

- The `16:12:00Z` raw-evidence-digest entry was inserted between earlier entries rather than appended at the end. Its timestamp and contents are valid; only its presentation position is wrong.
- The misplaced entry is retained unchanged to honor the append-only rule. This correction is the final authoritative ledger entry.

## 2026-08-23T16:12:00Z — raw evidence digest

- Generated `raw_output_sha256sums.txt` over all 140 files then present under `artifacts/raw/`.
- The manifest SHA-256 is `db7279e0fc0dd8f0a0080dc4e234d874b25217ef76c6d003af55644c7df26735`.
- This adds integrity metadata only. No raw output, locked source file, configuration, result threshold, or simulator parameter was edited.

## 2026-08-23T16:02:00Z — explicit SCons compiler selection was ineffective

- Attempt `reference_validation_gcc9_explicit_20260823T160200Z` is preserved as a third failed build attempt.
- Passing `CXX=/.../g++-9` on the SCons command line did not override the bare `g++` construction command created by this upstream `Sconstruct`; the raw log confirms the override was ignored.
- No source or build-script patch was applied.
- Decision A005: move the unmodified build into a pinned Ubuntu 20.04 Apptainer image, where the default `/usr/bin/g++` is GCC 9. This avoids monkey-patching SCons, altering the upstream `Sconstruct`, or installing host-global tools. The container definition is preserved as `pimsimulator_ubuntu20.def` and was fixed before reference-test or joint-gap observation.
- Local simulator patches remain **none**.

## 2026-08-23T16:00:17Z — preserved ineffective GCC 9 selection attempt

- Attempt `reference_validation_gcc9_20260823T160000Z` also failed during compilation with exit code 2 and is preserved intact.
- Diagnosis: SCons `Environment()` did not propagate the caller's prepended `PATH` into its construction environment; its raw build log invoked bare `g++` and reproduced the GCC 13 errors. The environment probe reporting GCC 9 was therefore not proof that SCons used GCC 9.
- A direct GCC 9.5.0 compile check of the previously failing translation unit succeeded without source changes.
- Decision A004: pass the absolute GCC 9 compiler path through SCons' `CXX=...` construction variable on the next clean attempt. This is explicit toolchain selection, not a source patch.
- Local simulator patches remain **none**.

## 2026-08-23T16:08:00Z — correction to ledger ordering

- The preceding `16:02:00Z` entry appears before the `16:00:17Z` entry. That is a presentation-order error only; the timestamps and preserved raw attempt identifiers are authoritative.
- The entries were not reordered because this ledger is append-only.

## 2026-08-23T16:09:00Z — pinned compatibility environment

- Built `dependencies/pimsimulator-ubuntu20.04.sif` from the preserved `pimsimulator_ubuntu20.def` without changing simulator source.
- The definition SHA-256 is `11a13fc5c76147030d78cf7db61d04f523650e0989987ea0ea364fb572ca49d1`; the image SHA-256 is `5e9455a5c0cfdf9e68c818473f66498bc1b7e911850f85e5b30a40091ec12db4`.
- The image uses Ubuntu 20.04, GCC 9.4.0, SCons 3.1.2, and GoogleTest 1.10.0. Exact package versions are recorded in `source_lock.json` and the successful attempt's `environment.txt`.
- Attempts `reference_validation_container_20260823T160500Z` and `reference_validation_container_notty_20260823T160600Z` failed before compilation because the restricted execution sandbox denied Apptainer user-namespace socket operations. These are infrastructure failures, not simulator validation evidence, and their raw logs are preserved.
- Assumption A006: running the already pinned image outside that restricted sandbox is permitted because it changes only the execution context, not source, configuration, compiler, dependencies, or simulator parameters.
- Local simulator patches remain **none**.

## 2026-08-23T16:10:00Z — reference validation gate failed; arm stopped

- Attempt `reference_validation_container_escalated_20260823T160700Z` built the unmodified locked source successfully (build exit code 0) and discovered the expected 11 reference tests (discovery exit code 0).
- Ten tests passed. `PIMBenchFixture.add` failed with exit code 1: simulated non-PIM cycle count 6651, simulated PIM cycle count 3349, ratio 1.985965967, while upstream requires a ratio strictly greater than 2.0.
- No threshold, timing, mapping, test dimension, source, or configuration was changed after observing the failure. The failed result was not reclassified as a pass.
- Gate G001 therefore failed. No representative Stage 1 workload was selected, no trace/mapping adapter was implemented, no Stage 1 simulator measurements or transaction traces were generated, no analytical model was calibrated or validated, and no joint-optimization sweep was launched.
- The ten passing test outputs remain preserved as partial diagnostic observations only. They are not an admissible HBM-PIM cost profile and must not be consumed by the joint optimizer.
- Decision A007: INT8 is not accepted as native supported execution. Although the configuration parser recognizes `INT8`, the fallback arithmetic operators in `Burst.h` operate on sixteen FP16 lanes. FP32 has explicit eight-lane arithmetic branches but was not covered by the locked FP16 reference suite. Both are therefore unvalidated for this arm; FP8, INT4, block scaling, and sparse execution are absent and `UNSUPPORTED`.
- Data rule D001 remains in force: directional JSON records sharing one physical `run_id` are reconstructed views, never independent measurements or statistical samples.
- Local simulator patches remain **none**.

## 2026-08-23T16:14:00Z — final presentation-order correction

- The `16:12:00Z` raw-digest entry and its `16:13:00Z` correction were both inserted earlier in the file by the ledger-writing procedure. Neither entry's substantive content changed, and neither action touched simulator source or raw evidence.
- This entry is physically appended and supersedes only the `16:13:00Z` sentence claiming to be the final entry. Chronological timestamps, run identifiers, and preserved artifacts remain authoritative.
