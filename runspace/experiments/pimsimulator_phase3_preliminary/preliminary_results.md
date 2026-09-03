# Phase-3 preliminary simulator results

## Result

The representative native FP16 kernels are feasible to map and model for the exact documented HBM2-PIM simulator configuration. The native adapter and analytical model passed their narrow preregistered gates, but the arm is stopped before joint optimization because full-input provenance, preregistration completeness, component attribution, workload mapping, and representation validity are not closed.

All values below are simulator cycles labeled `SIMULATED_PIM_HBM2`. The microsecond conversion uses the locked simulator `tCK=1 ns`; it is not host- or prototype-measured wall time.

| Workload | Initial residency | Kernel execution | Readback | Native total | At tCK=1 ns | Scope |
|---|---:|---:|---:|---:|---:|---|
| layer4.0 conv2 | 50,008 | 731,644 | 8,875 | 790,527 | 790.527 us | GEMV portion only |
| layer1.0 post-add ReLU | 1,837 | 2,185 | 1,836 | 5,858 | 5.858 us | Native end-to-end mapping |
| layer3.0 residual ADD | 1,069 | 1,390 | 556 | 3,015 | 3.015 us | Native end-to-end mapping |
| fc classifier | 22,486 | 6,606 | 181 | 29,273 | 29.273 us | Native portion only |
| global avgpool | — | — | — | `UNSUPPORTED` | — | No native reduction kernel |

Every supported workload had zero bit mismatches. Both repeats produced identical cycles and outputs.

The convolution number excludes im2col/patch packing and the cost of 376,320 host partial-sum additions. The classifier excludes the cost of 15,000 host partial-sum additions. Those operations are counted but uncosted; therefore neither native-only total is an end-to-end layer latency.

## Gate result

| Gate | Status | Evidence |
|---|---|---|
| Adapter correctness/determinism | Pass | 30/30 exits 0; zero bit mismatches; exact repeat determinism |
| Analytical versus simulator | Pass | 2.5573% MAPE; 4.9226% worst case |
| Limited local hash guard | Pass | Identified simulator, feasibility, diagnostic, shape, hybrid-JSON, and database hashes match |
| Full input provenance | Fail | Required L40S, H100, parametric, Stage-1-plan, checkpoint, and locked-split manifests not located |
| Preregistration completeness | Fail | Several required Phase-3A fields were absent before results |
| End-to-end mapping | Fail | Unvalidated im2col/packing and host reductions; avgpool unsupported |
| Representation support | Fail | Only dense FP16 is validated natively |
| Joint optimization | Not run | Required downstream gates did not pass |

Consequently, exhaustive joint, placement-first, representation-first, and both alternating procedures were not executed. Positive, zero, and negative joint gaps are all `NOT_EVALUATED`; this is not a zero-gap result and no regime claim is made.
