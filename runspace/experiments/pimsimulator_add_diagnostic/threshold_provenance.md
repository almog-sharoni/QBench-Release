# `PIMBenchFixture.add` threshold provenance

## Local source history

The strict threshold is present in two places at the locked commit:

- `src/tests/PIMBenchTestCases.cpp:45` calls `expectPIMBench(2.0)`.
- `src/tests/PIMBenchTestCases.h:295` tests `float(non_pim_cycle_) / pim_cycle_ > expected_perf_gain`.

`git blame` attributes both lines to the repository's initial commit, `b00e73a3ca1a421eb10a96b0d2f5a0f1ef973ed8`, authored 2022-11-21. That commit assigns `2.0` to GEMV, MUL, ADD, and ReLU. The commit message is only “Initial commit”; it contains no cycle table, platform qualification, confidence interval, or rationale for the strict value.

The threshold pickaxe history shows no later threshold edit through locked commit `3703d1f19c8f027360cc33a3243eb271e3bb6898`.

Commit `994d66bf7172903dd8d198dfe4313bcb4326738a` (2023-03-20/21) later changed `PIMKernel::parkOut()` from two transactions to iteration over all banks. Its stated rationale is that `parkOut()` runs after switching to SB mode, so all banks should be touched. The threshold was not changed in that commit. The current PIM ADD trace shows the final all-bank park-out drain extending through cycle 3,349.

This is evidence of threshold/code-history drift. It is not evidence that the old code had a particular cycle count, because this arm deliberately did not switch revisions merely to search for a passing outcome.

## Documentation and public tracker review

The [upstream README](https://github.com/SAITPublic/PIMSimulator) describes `PIMKernelFixture` as functionality testing and `PIMBenchFixture` as performance testing and documents how to invoke the tests. It does not publish the exact ADD reference cycles or a rationale for `> 2.0`.

The public branch/tag API snapshot found one `dev` branch at the locked commit and no tags. The public issue/pull-request snapshot and diffs contained no change to the ADD threshold and no cited 6,651/3,349 reference pair. Relevant context was preserved rather than treated as a substitute benchmark:

- [Issue #8](https://github.com/SAITPublic/PIMSimulator/issues/8) reports historical ADD/MUL functionality trouble; a maintainer attributes it to missing dummy CRF code and says the latest code fixed it. The locked functional run and exact-bit probe pass.
- [Issue #3](https://github.com/SAITPublic/PIMSimulator/issues/3) explains that PIM operations consume `tCCD_L`, repeated per operation.
- [Issue #10](https://github.com/SAITPublic/PIMSimulator/issues/10) cautions that the non-PIM performance path is an oracle-style sequence of memory accesses and omits host compute overhead, especially problematic for high-locality operations.
- [Issue #15](https://github.com/SAITPublic/PIMSimulator/issues/15) and [issue #16](https://github.com/SAITPublic/PIMSimulator/issues/16) discuss timing interpretation; #16 states that PIM operation and operand fetch occur within modeled DRAM timing constraints.

None of these sources documents a different supported configuration for `PIMBenchFixture.add`, an exact expected cycle count, or justification for retaining the strict threshold after the all-bank `parkOut()` change.

## Scope of the provenance conclusion

The conclusion is limited to the locked repository history and public GitHub material captured on 2026-08-23. It does not assert that unpublished Samsung environments or intentions do not exist. It establishes that the available upstream record supports a brittle performance expectation explanation and does not support a functional, nondeterministic, or configuration-mismatch explanation for this run.

Raw blame, commit, file-history, pickaxe, documentation-search, branch/tag, issue, comment, pull-request, and diff records are preserved under `provenance/`.
