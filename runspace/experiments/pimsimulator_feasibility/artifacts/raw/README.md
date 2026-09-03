# Raw evidence

Each `reference_validation_*` directory is immutable run evidence with captured source identity, environment, configurations, build stdout/stderr/exit code, and—where reached—test discovery plus per-test commands, stdout, stderr, and exit codes.

The authoritative complete unmodified-source attempt is `reference_validation_container_escalated_20260823T160700Z`. Its build succeeded, 10 tests passed, and `PIMBenchFixture.add` failed. Earlier directories are preserved build or execution-environment failures and are not simulator measurement evidence.

No Stage 1 workload simulation or transaction trace was produced because the reference-validation stop gate failed.
