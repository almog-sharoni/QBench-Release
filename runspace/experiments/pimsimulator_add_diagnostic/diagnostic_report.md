# PIMSimulator ADD post-validation diagnostic report

## Outcome

The sole classification is **`UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`**.

At locked commit `3703d1f19c8f027360cc33a3243eb271e3bb6898`, the unmodified test compares `float(non_pim_cycles) / pim_cycles > 2.0`. A clean build repeatedly produced 6,651 non-PIM cycles and 3,349 PIM cycles. The exact real-valued ratio is 1.985965959988056; the assertion prints the single-precision value 1.9859659671783447 and correctly fails.

This classification does not turn the failure into a pass. The original frozen Phase-2 result remains failed and byte-identical. The observed PIM cost remains 3,349 cycles; it is never replaced by the threshold-implied maximum of 3,325 cycles.

## Why this is not a functional failure

The unmodified built-in `PIMKernelFixture.add` test exited 0 and reported 1,048,576 passed values and zero failed values. Because its comparator permits numerical tolerances, an external read-only diagnostic harness also compared every result as an exact 16-bit FP16 bit pattern:

- 1,048,576 elements / 65,536 bursts compared;
- zero mismatches against the locked NumPy reference output;
- zero mismatches against independently recomputed FP16 input0 + input1;
- no first mismatch in either comparison.

The performance test's non-PIM branch models traffic only—two input reads and one output write—and does not calculate or emit an output tensor. Therefore the available non-PIM/host numerical references are (1) the committed NumPy result and (2) a fresh element-wise FP16 host recomputation from the two committed inputs. The PIM output matched both bit-for-bit.

The harness source is separate from upstream and links the unchanged locked static simulator library. Its 9,626 cycles include preload and result-readback work and therefore are functionality-path evidence only, not a replacement performance measurement.

## Why this is not nondeterminism

Five serial executions used the same binary SHA-256 `e348be1c030166f3ed6eee7f2632f996f21472c4aecdb2620797b4920944ae9b`. Every execution returned:

- non-PIM cycles: 6,651;
- PIM cycles: 3,349;
- printed speedup: 1.98597;
- assertion exit code: 1.

The raw stdout hashes differ only because GoogleTest prints wall-clock durations. After replacing only those duration fields with a fixed token, all five stdout streams hash to `f8bce28d0a574cafc67ddb46787adfb2af327250eb815f93fe31ab8525e597f9`.

## Why this is not a configuration mismatch

The benchmark constructor itself hard-codes `ini/HBM2_samsung_2M_16B_x64.ini`, `system_hbm_64ch.ini`, 64 PIM channels, and one PIM rank. The clean run used those exact files from the locked source archive. Their hashes were captured before execution. The fresh clone had no local patches and no dirty files.

No upstream document, branch, tag, issue, pull request, or commit inspected in this arm identifies a different supported configuration for this assertion or provides a different reference cycle count. Absence from the inspected public record is not proof that no private or historical environment existed; it means a configuration-mismatch explanation is not evidenced.

## Performance-threshold deviation

For a strict `6651 / pim_cycles > 2.0` comparison, the largest passing integer PIM cycle count is 3,325. The observed 3,349 is 24 cycles higher, a 0.7218045% excess over that maximum; the speedup is 0.7017020% below 2.0.

A logging-only trace probe retained the byte-identical simulator binary and changed only output controls. In the PIM phase:

- the final all-bank `END_PARK_OUT_BAR` activation is issued at cycle 3,312;
- later all-channel park-out reads/precharges include the bank-15 read at 3,326 and precharges at 3,327, 3,330, and 3,345;
- the transaction queue drains at the reported cycle 3,349.

Thus the 24 cycles beyond the maximum passing cycle 3,325 lie entirely in the final all-bank park-out drain tail. This is not a proposal to remove those cycles.

The `> 2.0` calls and helper entered the repository in initial commit `b00e73a3ca1a421eb10a96b0d2f5a0f1ef973ed8`. Later commit `994d66bf7172903dd8d198dfe4313bcb4326738a` changed `parkOut()` from touching two banks to touching all banks, explaining that all banks should be touched after switching to SB mode. The threshold itself was not updated. This history supplies a concrete provenance-consistent explanation for a small, deterministic performance expectation miss without making an unsupported claim about the exact cycle count on the initial commit.

## Clean rebuild and full-suite result

The locked source was exported into a newly created build directory and compiled in the pinned Ubuntu 20.04 container with g++ 9.4.0 and SCons 3.1.2-2. Build exit code was 0. The complete unmodified suite ran 11 tests: 10 passed and only `PIMBenchFixture.add` failed, with suite exit code 1.

## Recommendation and stop condition

The simulator may proceed only to a future, separately reviewed and preregistered validation stage under the transparent label:

`VALIDATED_WITH_UPSTREAM_PERFORMANCE_THRESHOLD_DEVIATION`

That stage must retain 3,349 cycles, the failing assertion, the `SIMULATED_PIM_HBM2` evidence class, and all original Phase-1 validity boundaries. It must not claim an upstream test pass or physical PIM measurement.

This arm stops here. No workload adapter, analytical-model integration, simulator parameter tuning, or joint-optimization sweep was launched.
