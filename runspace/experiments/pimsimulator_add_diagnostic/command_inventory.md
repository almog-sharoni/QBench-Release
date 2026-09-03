# Diagnostic command inventory

This inventory points to the exact executable command records. Exploratory read-only inspection commands are not experimental runs.

## Source and frozen-input checks

- Fresh source acquisition: `git clone --branch dev https://github.com/SAITPublic/PIMSimulator.git source/PIMSimulator`
- Source lock checks, clean archive export, environment capture, build, tests, and output hashing: `exact_reproduction.sh`
- Frozen Phase-2 recheck: `cd ../pimsimulator_feasibility && sha256sum --quiet -c ../pimsimulator_add_diagnostic/frozen_phase2_sha256sums.txt`

## Clean simulator run

The canonical commands are preserved under `runs/add_diagnostic_20260823T163000Z/raw/`:

- build: `build/command.txt`
- functionality: `functional_add/command.txt`
- repeats: each `performance_repeats/repeat_N/command.txt`
- complete suite: `full_suite/command.txt`

The clean runner records the exact container path and hashes before executing these commands.

## Exact-bit diagnostic

All three compile attempts, including the two failed path-resolution attempts, are under `runs/add_diagnostic_20260823T163000Z/build/exact_compare/`:

- first: `build_command.txt`, `build_stdout.txt`, `build_stderr.txt`
- second: `compile_attempt_2/command.txt`, `stdout.txt`, `stderr.txt`
- successful third: `compile_attempt_3/command.txt`, `stdout.txt`, `stderr.txt`
- execution: `run_command.txt`, `stdout.txt`, `stderr.txt`, `exit_code.txt`

The third command record redacts the machine-specific diagnostic root and container path as `DIAGNOSTIC_ARM` and `CONTAINER`; those exact locked values are recorded in `exact_reproduction.sh` and the run's `raw/environment.txt`.

## Trace diagnostic

- command: `runs/add_diagnostic_20260823T163000Z/trace_probe/command.txt`
- changed logging controls: `trace_probe/config_diff.patch`
- stdout/stderr/exit: `trace_probe/stdout.txt`, `stderr.txt`, `exit_code.txt`
- before-run executable identity: `trace_probe/sim_sha256_before.txt`

## Provenance

Exact local git commands are paired with their outputs under `provenance/raw/*_command.txt`. Public GitHub API paths and corresponding preserved response filenames are listed in `provenance/github_api/commands.txt`.
