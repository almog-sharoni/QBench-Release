# Limitations and unsupported claims

## Blocking limitation

The unmodified built-in suite did not fully reproduce. `PIMBenchFixture.add` produced 6,651 simulated non-PIM cycles and 3,349 simulated PIM cycles (1.985965967×), failing the upstream requirement of strictly greater than 2×. Under the preregistered stop rule, no cost-profile or analytical-validation claim follows from this arm.

## Architecture boundary

The source models its documented HBM2 PIM organization and timing through a DRAMSim2-derived cycle model. It does not establish behavior for:

- a generic processing-near-memory device;
- LPDDR5X-PIM;
- a fabricated or instrumented physical PIM prototype;
- L40S or H100 memory/compute behavior.

The simulator's “non-PIM” path is still a simulated HBM2 path. It is not a measured host/GPU baseline.

## Precision boundary

- FP16 is the configured built-in reference path, with sixteen FP16 values per 256-bit burst.
- FP32 code contains explicit eight-lane arithmetic branches, but this arm did not validate them.
- `INT8` appears in configuration parsing and changes nominal bytes per datum, but the PIMBlock fallback arithmetic delegates to `BurstType` operators that iterate over FP16 lanes. Native INT8 arithmetic and throughput are therefore unsupported here.
- FP8, INT4, quantization block scaling, scale metadata handling, and low-precision transaction packing are not implemented and validated.
- No lower traffic or compute-cycle count may be inferred from nominal bit width alone.

## Operation and mapping boundary

Built-in mappings cover GEMV/GEMV-tree, element-wise ADD/MUL, and ReLU. There is no native convolution mapping. A defensible ResNet-50 study would still need a reviewed adapter that accounts for tensor layout, tiling, padding, partial sums, residency, all host/PIM transfers, CRF programming, mode transitions, barriers, synchronization, result readback, conversion/packing, quantization scales, and sparse indices. That adapter was intentionally not implemented after the gate failure.

The built-in performance tests do not by themselves prove arbitrary-shape validity, end-to-end ResNet-50 latency, or optimizer suitability.

## Missing downstream evidence

- No preregistered ResNet-50 shape subset exists in this arm.
- No Stage 1 transaction trace or cycle measurement exists.
- No analytical metadata/conversion cost was added.
- No existing Stage 1 analytical PIM latency implementation was located in the repository during the preflight inspection; in any case comparison was blocked by the reference gate.
- No calibration/development split or locked validation split exists.
- No absolute error, relative error, or error decomposition was computed.
- No exhaustive, placement-first, representation-first, or alternating joint optimization was run.
- No positive, zero, or negative joint-gap result was produced.

Empty or not-run fields must not be interpreted as zero latency, zero error, or zero joint gap.

## Conditions for any future arm

Continuation requires review and a new post-result preregistration that preserves this failed arm. It must state whether it uses a newer upstream commit, a declared patch, or a changed environment; lock all such changes before Stage 1 joint-gap observation; rerun the complete reference suite; and keep calibration cases separate from locked validation cases. A physical LPDDR5X-PIM validation claim would require independent physical evidence outside this simulator.
