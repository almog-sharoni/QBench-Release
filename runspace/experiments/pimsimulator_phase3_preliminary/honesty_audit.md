# Phase-3 honesty audit

## Audit result

The arm complies with the preregistered evidence and stopping rules.

- Every simulator-derived value is labeled `SIMULATED_PIM_HBM2`; none is labeled host-, PIM-, or prototype-measured.
- The accepted upstream test remains a failure: exit code 1, strict `> 2.0`, 6,651 non-PIM cycles, 3,349 PIM cycles. The diagnostic record of the 24-cycle final all-bank park-out drain remains unchanged.
- The exact frozen feasibility and diagnostic bundle hashes are referenced by the new manifest and rechecked by reproduction and analysis.
- No frozen feasibility or diagnostic file was intentionally edited. The ledger discloses the transient creation and exact removal of empty directories during preflight; the diagnostic digest remained identical afterward.
- The simulator source was not silently patched. Adapter code is external. The only trace configuration change enables logging and disables summary printing in a copied run-local source tree; its diff is preserved.
- Initial residency, kernel execution, result readback, logical traffic, padded traffic, and external work are separated.
- PIM mode entry, CRF programming, synchronization/barriers, and result readback are included in native cycle capture.
- FP16 has no quantization-scale metadata. Low-precision scale, packing, and sparse-index costs are not invented.
- FP8, INT8, INT4, block scaling, packed sub-16-bit traffic, and sparse execution receive no latency or throughput benefit.
- Avgpool is `UNSUPPORTED`, not zero-cost.
- Convolution im2col/packing and GEMV host reductions are counted but uncosted; native totals are not presented as end-to-end layer latency.
- Development data alone fitted the analytical model. Locked validation was not used for calibration, and no threshold/features changed after observing results.
- Repeated simulator executions establish determinism only and are not independent samples.
- Directional quality JSON views sharing a physical `run_id` remain visible but are not independent measurements. Phase 3 uses zero such records as PIM cost samples.
- No L40S/H100 throughput or bandwidth enters this cost profile. No frozen quantization-quality result was reinterpreted.
- No result is generalized to LPDDR5X-PIM, generic PNM, or a physical prototype.
- No joint optimizer ran. Joint gaps are `NOT_EVALUATED`, never coerced to zero.

The stopped result is intentional: passing native-kernel feasibility does not close the end-to-end workload or representation validity boundaries.

## Full-specification correction

The earlier limited provenance gate checked only the feasibility bundle, diagnostic bundle, shape JSON, hybrid-quality JSON, and database. The subsequently supplied complete Phase-3 specification also requires frozen L40S, H100, parametric, Stage-1-plan, checkpoint, and locked-final-split inputs. These were not located, so the authoritative full-input provenance decision is `FAIL`. The earlier narrow `phase3_completion_manifest.json` is superseded by `phase3_full_spec_completion_manifest.json`; it is retained rather than silently erased.

The initial preregistration also omitted required pre-result fields. They are reported as missing in `preregistration_compliance_audit.csv` and are not retroactively declared preregistered. The locked 48,000-image final-evaluation split was not opened, decoded, evaluated, or enumerated.

The host-quality JSON does not itself prove whether a record is `MEASURED_HOST_L40S` or `EMULATED_HOST_L40S`. `physical_run_provenance.csv` therefore uses `UNRESOLVED_HOST_PROVENANCE`; no stronger device label is invented.
