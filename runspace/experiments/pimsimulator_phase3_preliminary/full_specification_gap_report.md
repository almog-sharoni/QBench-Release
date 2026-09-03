# Full Phase-3 specification gap report

## Decision

`STOPPED_FULL_SPEC_PROVENANCE_AND_PREREGISTRATION_FAILED`.

The native simulator adapter and the preregistered native analytical-validation test remain valid within their narrow scope. They do not authorize Stage-1 workload integration because the later-provided full Phase-3 specification exposes missing pre-result commitments and missing frozen inputs.

The following required frozen artifacts could not be located in the current workspace and therefore have no defensible hash in this arm:

- the L40S measurement bundle;
- the H100 comparison bundle;
- the parametric boundary-cost bundle;
- the frozen Stage-1 plan;
- the locked 48,000-image final-evaluation split manifest;
- the exact resolved ResNet-50 checkpoint or checkpoint hash.

The ImageNet final-evaluation data was not listed, opened, decoded, or evaluated. A missing split manifest was not reconstructed from the data.

The original preregistration also did not define residency modes, sensitivity ranges, quality constraints, fixed-point convergence, tie-breaking, or the required figure/report inventory before simulator results were observed. These omissions cannot be repaired retroactively. `preregistration_compliance_audit.csv` records each field and its evidence.

## Additional mapping boundary

The raw traces preserve mode, CRF, compute, and park commands, but the workload adapter does not separately attribute cycles to mode entry/exit, CRF programming, PIM memory, PIM compute, synchronization, or each workload's parkOut contribution. Those components are included in aggregate simulator phases, not independently validated cost terms. The known 24-cycle final all-bank parkOut diagnostic remains visible and unchanged, but it is not assigned as a per-layer analytical constant.

Consequently, the host–PIM breakdown is incomplete, the full provenance gate fails, and optimization remains forbidden. The defense artifacts render unavailable values as blank or `NOT_EVALUATED`; no unavailable quantity is encoded as zero.

No coupled, separable, mixed, positive-gap, zero-gap, or negative-gap regime claim is made. The permitted conclusion is:

> The current simulator coverage, workload mapping, and full-input provenance are insufficient for a coupled-versus-separable regime claim.
