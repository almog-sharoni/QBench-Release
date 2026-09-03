# Analytical versus PIMSimulator validation

## Decision

`PASS` for the preregistered native-kernel analytical-validation gate.

- Locked validation cases: 4
- Total-cycle MAPE: 2.5573% (limit: 10%)
- Worst locked absolute relative error: 4.9226% (per-case limit: 20%)
- Median locked absolute relative error: 2.5826%
- Calibration membership: 11 development cases only
- Locked validation used in fitting: no
- Post-result feature or threshold tuning: none

Passing this gate validates the fitted cycle model only for the mapped native FP16 ADD, ReLU, and GEMV cases in this configuration. It does not validate a full ResNet-50 latency or another numerical representation.

## Locked results

| Case | Kernel | Simulator cycles | Analytical cycles | Signed error | Absolute relative error |
|---|---:|---:|---:|---:|---:|
| layer3.0 residual ADD | ADD | 3,015 | 2,866.58 | -148.42 | 4.9226% |
| layer1.0 post-add ReLU | ReLU | 5,858 | 6,131.33 | +273.33 | 4.6660% |
| fc classifier | GEMV | 29,273 | 29,314.39 | +41.39 | 0.1414% |
| layer4.0 conv2 GEMV portion | GEMV | 790,527 | 794,473.48 | +3,946.48 | 0.4992% |

`analytical_vs_simulator_validation.csv` contains phase-level error decomposition for initial residency, kernel execution, and result readback, as well as development-case residuals. `analytical_model_coefficients.json` preserves the exact features, coefficients, ranks, observation counts, and singular values.

The specification-name artifact `analytical_vs_simulated_validation.csv` adds provenance labels and a descriptive tensor-size group. `analytical_validation_statistics.csv` reports mean, median, and worst error overall, by operation, and by tensor-size band. Tensor-size bands were added after capture for descriptive reporting only and do not participate in the frozen acceptance test. `analytical_vs_simulated_latency.pdf` plots every development and locked case against y=x; no outlier is removed.

The largest total relative error is the ADD validation case. For the convolution mapping, the analytical overestimate decomposes into +26.63 residency cycles, +6,075.71 execution cycles, and -2,155.87 readback cycles. These figures apply only to the native GEMV portion; im2col/packing and host reduction are deliberately absent from both totals.

## Interpretation boundary

The model is descriptive interpolation/extrapolation over the preregistered shapes, not a universal HBM2-PIM performance law. There are only three development points per element-wise kernel and five GEMV development points. Ordinary least squares yields some nonphysical intercepts or individual phase coefficients; only the preregistered aggregate validation criterion passed. Coefficients must not be transferred to LPDDR5X-PIM, generic PNM, low precision, sparse execution, or a physical prototype.
