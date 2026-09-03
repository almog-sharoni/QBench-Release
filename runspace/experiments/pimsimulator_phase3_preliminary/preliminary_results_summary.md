# Phase-3 preliminary results summary

## Conclusion

The current simulator coverage, workload mapping, preregistration, and full-input provenance are insufficient for a coupled-versus-separable regime claim.

PIMSimulator does provide reproducible native dense-FP16 observations for the selected ADD, ReLU, and GEMV mappings at the exact locked Samsung HBM2 configuration. This narrow feasibility result does not close the Stage-1 host–PIM or representation space, so no joint or fixed-point sequential optimizer was run.

## Evidence layers

| Evidence question | Result | Provenance |
|---|---|---|
| What was measured/emulated on the host? | The hybrid JSON exposes 83 ResNet-50 directional quality records representing 76 physical runs. Its file does not establish the required L40S device provenance, so these records are not consumed as admissible Stage-1 quality/cost samples. | `UNRESOLVED_HOST_PROVENANCE` |
| What shapes were obtained? | Five preregistered ResNet-50 representatives cover compute-heavy GEMV lowering, large weights, large activations, classifier GEMV, element-wise ADD/ReLU, and an unsupported avgpool control. | `TRACE_DERIVED`, with unresolved checkpoint hash |
| What was simulated? | 30/30 native adapter executions exited zero, were bit-exact, and reproduced identical cycles. | `SIMULATED_PIM_HBM2` |
| Did the analytical model match? | Yes in its narrow native scope: four locked cases, 2.5573% mean absolute relative error, 2.5826% median, and 4.9226% worst case. | `CALIBRATED_HYBRID_MODEL` versus `SIMULATED_PIM_HBM2` |
| Is end-to-end host–PIM latency available? | No. Im2col/packing and host reductions are uncosted, avgpool is unsupported, and several native phases are not separately cycle-attributed. | Unavailable values remain blank |
| Did a joint-over-fixed-point gap appear? | `NOT_EVALUATED`. This is not a zero gap. | No optimizer output exists |
| What mechanism caused a gap? | None was evaluated. Every ablation is deferred under the stop rule. | `mechanism_ablation.csv` |

## Native preliminary observations

| Workload | Native cycles | Scope |
|---|---:|---|
| layer3.0 residual ADD | 3,015 | Native end-to-end mapping with reported tile padding |
| layer1.0 post-add ReLU | 5,858 | Native end-to-end mapping with reported tile padding |
| fc classifier | 29,273 | Native GEMV portion; 15,000 host reductions uncosted |
| layer4.0 conv2 | 790,527 | Native GEMV portion; im2col and 376,320 host reductions uncosted |
| global avgpool | `UNSUPPORTED` | No native pooling/general-reduction kernel |

All cycle values retain the label `SIMULATED_PIM_HBM2`. Conversion at simulator `tCK=1 ns` is simulated time, not host or prototype wall time.

## Why the arm stopped

- Required frozen L40S, H100, parametric, Stage-1-plan, locked-split-manifest, and exact-checkpoint hashes were not available in the workspace.
- The locked 48,000-image final-evaluation split was not opened, decoded, evaluated, or enumerated.
- Residency modes, sensitivity ranges, quality constraints, fixed-point convergence, tie-breaking, and required figures were not all preregistered before results.
- Dense FP16 is the only validated PIM representation. All low-precision directional records are `UNSUPPORTED_PIM_FORMAT` for PIM costing.
- Host boundary costs and required end-to-end operations are incomplete.

The upstream ADD deviation remains unchanged and visible: exit code 1, strict speedup `> 2.0`, 6,651 non-PIM cycles, 3,349 PIM cycles, and the identified 24-cycle final all-bank parkOut drain. It was not converted into a passing upstream test.

## Presentation artifacts

- `phase3_preliminary_defense_slide.pdf` — one-slide defense summary
- `speaker_script.md` — short presentation script
- `analytical_vs_simulated_latency.pdf` — all points with y=x reference
- `host_pim_cost_breakdown.pdf` — native aggregates plus explicit unavailable components
- `joint_gap_regime_map.pdf` — stopped/no-regime result, not a zero-gap map
- `preliminary_results_table.csv`, `physical_run_provenance.csv`, `mechanism_ablation.csv`, and `provenance_and_claim_ledger.csv` — machine-readable evidence
