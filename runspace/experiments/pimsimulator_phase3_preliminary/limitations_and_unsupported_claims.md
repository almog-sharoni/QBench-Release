# Limitations and unsupported claims

## Validated scope

This arm supports only the exact PIMSimulator commit and locked Samsung HBM2 configuration documented in `phase3_manifest.json`: 64 channels, one rank, native dense FP16, and the tested ADD, ReLU, and GEMV mappings. It establishes simulator feasibility and a small-shape analytical comparison—not hardware validation.

## Known limitations

- The workload subset has four supported native mappings and one deliberately unsupported control; it is not a complete ResNet-50 trace.
- Convolution uses sequential GEMV calls. Input im2col/patch creation and packing have no validated cycle cost.
- GEMV returns native partial sums. Host reduction operations are counted but their latency, traffic beyond readback, and host/PIM synchronization cost are not validated.
- Global adaptive average pooling has no supported native mapping.
- Native GEMV output tiling and element-wise 131,072-element tiling cause substantial padding, which is included and reported.
- Initial residency is explicit, but no system-level cache, overlap, contention, multi-layer reuse policy, or placement transition scheduler has been validated.
- The analytical model is fitted to 11 development shapes and checked on four locked shapes. It should not be extrapolated outside the covered mapping without new validation.
- Cycle-to-time conversion uses simulator tCK and is not a measurement of a deployed device.
- The study does not validate accuracy, numerical error, or conversion semantics for a whole model.

## Unsupported claims

This arm cannot support claims of:

- native or accelerated FP8, INT8, INT4, block-scaled, packed low-precision, or sparse execution;
- reduced traffic or compute cycles from quantization;
- validated quantization-scale or sparse-index metadata costs;
- end-to-end latency for the selected convolution, classifier, avgpool, or complete ResNet-50;
- a joint placement/representation optimum or any positive, zero, or negative joint gap;
- parity with or speedup over L40S, H100, another host, or another PIM/PNM device;
- a generic PNM, LPDDR5X-PIM, or physical Samsung prototype result;
- independent statistical replication from deterministic reruns or reconstructed directional JSON records.

Unsupported configurations must remain `UNSUPPORTED` until separately implemented, preregistered, and validated. Missing external costs must remain uncosted—not zero.

## Full-specification provenance limitations

- Exact frozen L40S, H100, parametric, Stage-1-plan, and locked-final-split manifest hashes were not found in the current workspace.
- The model configuration specifies torchvision `DEFAULT` weights but does not freeze the resolved enum or checkpoint SHA-256.
- Host-quality records cannot receive `MEASURED_HOST_L40S` or `EMULATED_HOST_L40S` solely from the available directional JSON.
- The initial Phase-3 preregistration did not freeze all required sensitivity, quality, residency, convergence, tie-breaking, classification, and presentation fields.
- Simulator traces contain mode/CRF/compute/park commands, but their cycles are not separately attributable to every requested host–PIM cost component.

These are stopping failures, not minor caveats. They prohibit optimizer integration and any regime claim.
