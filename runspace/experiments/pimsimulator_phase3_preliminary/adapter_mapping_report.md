# Phase-3 workload adapter report

## Outcome

The external adapter passed for the locked HBM2, 64-channel, one-rank, FP16 configuration at PIMSimulator commit `3703d1f19c8f027360cc33a3243eb271e3bb6898`. It does not patch or extend the simulator ISA.

All 15 cases were run twice. Every execution exited 0, reported zero bit mismatches, and produced identical phase cycles and output hashes across repeats. The two executions are determinism checks, not independent experimental samples.

## Mapping boundary

| Workload | Native mapping | Padding/residency rule | Outside-PIM work | End-to-end status |
|---|---|---|---|---|
| layer3.0 residual ADD | Native all-bank ADD | 200,704 logical elements padded to 262,144 | None | Available |
| layer1.0 post-add ReLU | Native all-bank ReLU | 802,816 logical elements padded to 917,504 | None | Available |
| fc classifier | One native batch-1 GEMV | K padded to 128-value units; output uses the native 4,096-output tile | 15,000 partial-sum additions counted, no validated host-cycle cost | Native portion only |
| layer4.0 conv2 | 49 sequential batch-1 GEMVs with weights retained | M=512, K=4,608; 37,748,736 resident padded weight bytes | im2col/patch packing plus 376,320 partial-sum additions counted, no validated host-cycle cost | Native portion only |
| global avgpool | None | None | Unsupported reduction | `UNSUPPORTED` |

The measured phases are initial residency, PIM kernel execution (including mode entry, CRF programming, barriers, and synchronization), and result readback. Traffic fields separately expose logical and padded bytes. FP16 carries no quantization-scale metadata; sparse-index metadata is not applicable because every mapped case is dense.

Low-precision traffic and compute were not reduced. FP8, INT8, INT4, block-scaled, and sparse mappings remain `UNSUPPORTED`.

## Trace evidence

Raw transaction traces are preserved under `raw/phase3_preliminary_20260823T165500Z/traces/` and summarized in `transaction_trace_summary.csv`. The summary records compressed and uncompressed sizes, command counts, read/write/activate/precharge counts, tagged mode/CRF/compute commands, maximum cycle, trace SHA-256, and presence of the adapter result marker.

The trace run uses only a logging copy of `system_hbm_64ch.ini`. Its exact logging-only diff is preserved as `traces/config_diff.patch`; cost-bearing simulator parameters are unchanged.
