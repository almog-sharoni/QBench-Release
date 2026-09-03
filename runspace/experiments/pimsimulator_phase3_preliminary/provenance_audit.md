# Phase-3 provenance audit

## Decision

`PASS` for the limited local hash guard; `FAIL` for the subsequently supplied full-input provenance requirement.

| Input | Locked SHA-256 | Observed SHA-256 | Result |
|---|---|---|---|
| Frozen feasibility bundle | `8f0709f39267e79414cf9b7c9b2bde675b42531ffa45d90c0c71b730ca50f97b` | same | Match |
| Frozen ADD-diagnostic bundle | `6d64351133f952431d478162f644b0ddab6b5a3fb3fca40190d0fb5dd7fc8e96` | same | Match |
| ResNet-50 shape source | `6a6a4bec2b90a791b92023382a5fbb45354a0c0e629e6c84164ee44b809fae2d` | same | Match |
| Hybrid-quality JSON | `942093102afd8a850bfd790f41adc6c6617486379518dd5ee00e4ae056102501` | same | Match |
| Runs database | `401a06b517765a9811a4f489ba9b7dde58a821dc40bb100d94ca27ff8660ca2b` | same | Match |

The bundle digest algorithm is defined exactly in `phase3_manifest.json`. It includes every regular file, including nested `.git` files. The manifest also records exact hashes for the frozen source-lock, prior reports, simulator archive, and container.

The simulator source is upstream commit `3703d1f19c8f027360cc33a3243eb271e3bb6898` on `dev`; the archived source SHA-256 is `803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392`. Formal capture exported this commit into a new run directory, built it cleanly, and compiled an external adapter. No upstream patch was applied.

The raw run has its own `artifact_sha256sums.txt`, compiler/dependency record, build commands, binaries, configuration copies, exit codes, stdout, stderr, and trace hashes. The exact reproduction script refuses to overwrite an existing run and stops on any lock, build, functional, preserved-deviation, adapter, or trace failure.

## Quality-record accounting

The hybrid-quality source contains 83 directional records representing 76 unique physical `run_id` values. Seven physical IDs appear in both directions; eight records are explicitly reconstructed views. These records remain visible in `provenance/quality_run_accounting.json`.

Phase 3 consumed zero of them as independent statistical samples and assigned no PIM costs from them. Multiple directional views of the same `run_id` are never counted as independent measurements. Their sub-16-bit representations are unsupported by the validated PIM mapping.

## Full-input audit

`frozen_input_provenance.csv` is authoritative for the broader requirement. The frozen L40S, H100, parametric, Stage-1-plan, exact-checkpoint, and locked-48,000-image-split manifests were not located. Their hashes are null, not invented. This changes the integration decision to `PROVENANCE_GATE_FAILURE` even though every originally tracked local hash still matches.

The final-evaluation dataset was not enumerated or opened. `locked_final_split_guard.json` records zero opened, decoded, and evaluated samples.
