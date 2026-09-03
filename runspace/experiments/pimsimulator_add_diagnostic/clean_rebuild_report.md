# Clean rebuild report

## Locked inputs

- Upstream: `https://github.com/SAITPublic/PIMSimulator.git`
- Branch: `dev`
- Commit: `3703d1f19c8f027360cc33a3243eb271e3bb6898`
- `git archive` SHA-256: `803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392`
- Fresh clone status before export/build: clean
- Pinned container SHA-256: `5e9455a5c0cfdf9e68c818473f66498bc1b7e911850f85e5b30a40091ec12db4`

The build used a new `git archive` export at `runs/add_diagnostic_20260823T163000Z/build/source`. It did not reuse the prior Phase-2 build products and did not build in or alter the fresh source checkout.

## Toolchain

- Container OS: Ubuntu 20.04
- Compiler: `g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0`
- SCons package: `3.1.2-2`
- googletest/libgtest-dev: `1.10.0-2`
- libc6-dev: `2.31-0ubuntu9.18`
- Upstream compiler flags: `-g -O2 -std=c++14 -Wall -Wno-reorder -Wno-sign-compare` (plus position-independent-code flags for shared objects)

The exact package query and emitted compiler commands are preserved in `runs/add_diagnostic_20260823T163000Z/raw/environment.txt` and `raw/build/compiler_commands.txt`.

## Build result

- Command: `apptainer exec --userns <pinned-container> scons`
- Exit code: 0
- Built simulator SHA-256: `e348be1c030166f3ed6eee7f2632f996f21472c4aecdb2620797b4920944ae9b`
- Local upstream patches: none

Raw build stdout, stderr, command, exit code, compiler commands, and binary hash are preserved under `runs/add_diagnostic_20260823T163000Z/raw/build/`.

## Test results

| Test scope | Exit | Result |
|---|---:|---|
| `PIMKernelFixture.add` | 0 | 1,048,576 passed, 0 failed |
| Five `PIMBenchFixture.add` repeats | 1 each | 6,651 / 3,349 cycles each; strict ratio assertion failed each time |
| Complete upstream suite | 1 | 10 passed; only `PIMBenchFixture.add` failed |

The suite failure is preserved, not suppressed. The complete suite stdout and stderr are under `raw/full_suite/`.

## Supplemental diagnostic build

The external exact-bit harness is not an upstream patch and is not part of the clean simulator binary. Its first two compile attempts failed because the container could not resolve the host source path; both failed attempts are preserved. A third attempt explicitly bound the diagnostic arm, linked against the unchanged clean-build static library, and succeeded. The harness returned zero bit mismatches and exit code 0.

The logging-only trace probe copied the clean exported tree. Its simulator binary remained byte-identical; only three output-control configuration values changed, as recorded in `trace_probe/config_diff.patch`.
