# PIMSimulator feasibility and reference-validation report

## Outcome

**FAIL — stopped at the unmodified reference-test gate.** The locked simulator builds in the pinned Ubuntu 20.04 environment, but its complete built-in reference set does not reproduce: 10 of 11 required tests pass and `PIMBenchFixture.add` fails. Consequently, this arm does not provide a defensible Stage 1 HBM-PIM cost profile, and integration into the joint optimizer is not authorized.

This conclusion does not modify or reinterpret any frozen L40S, H100, quantization-quality, or parametric result bundle.

## Source and environment

- Upstream: `https://github.com/SAITPublic/PIMSimulator.git`
- Branch and commit: `dev` at `3703d1f19c8f027360cc33a3243eb271e3bb6898`
- Source archive SHA-256: `803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392`
- Source patches: none
- Clean compatibility environment: Ubuntu 20.04 Apptainer image, GCC 9.4.0, SCons 3.1.2, GoogleTest 1.10.0
- Image SHA-256: `5e9455a5c0cfdf9e68c818473f66498bc1b7e911850f85e5b30a40091ec12db4`
- Build command: `apptainer exec --userns dependencies/pimsimulator-ubuntu20.04.sif scons`
- Build exit code: 0
- Test-discovery exit code: 0
- Locked upstream working tree before and after validation: clean

The initial GCC 13 failure, two ineffective GCC 9 selection attempts, and two sandbox-blocked container invocations are all preserved. None changed simulator source. Exact versions, hashes, commands, stdout, stderr, configurations, and exit codes are in `source_lock.json` and `artifacts/raw/`.

## Unmodified test results

All numeric observations below are **SIMULATED_PIM_HBM2**. They are simulator-native outputs, not host measurements, physical PIM measurements, prototype measurements, or analytical additions.

| Test | Upstream shape | Native observation | Exit | Gate result |
|---|---:|---:|---:|---|
| HBM read bandwidth | 8,388,608 bytes | 231 GB/s | 0 | PASS |
| HBM write bandwidth | 16,777,216 bytes | 243 GB/s | 0 | PASS |
| GEMV functional | weight 4096×1024, input 1024, output 4096 | 4096 passed, 0 failed | 0 | PASS |
| GEMV tree functional | weight 4096×1024, input 1024, output 4096 | 4096 passed, 0 failed | 0 | PASS |
| ADD functional | 1,048,576 elements | 1,048,576 passed, 0 failed | 0 | PASS |
| MUL functional | 1,048,576 elements | 1,048,576 passed, 0 failed | 0 | PASS |
| ReLU functional | 1,048,576 elements | 1,048,576 passed, 0 failed | 0 | PASS |
| GEMV performance | 4096×4096 | non-PIM 36,082 cycles; PIM 13,166; 2.74054× | 0 | PASS |
| ADD performance | 1,048,576 elements | non-PIM 6,651 cycles; PIM 3,349; 1.985965967× | 1 | **FAIL** |
| MUL performance | 2,097,152 elements | non-PIM 13,255 cycles; PIM 5,926; 2.23675× | 0 | PASS |
| ReLU performance | 4,194,304 elements | non-PIM 17,504 cycles; PIM 7,665; 2.28363× | 0 | PASS |

The ADD failure is the upstream assertion that `non_pim_cycle / pim_cycle` be strictly greater than 2.0. The observed ratio is below that threshold. It was not rounded up, patched, or waived.

## Validity boundary discovered before stopping

PIMSimulator documents a cycle-accurate model based on DRAMSim2 for its HBM2 PIM architecture. It is not treated here as a generic PNM device, LPDDR5X-PIM, or validation of a physical prototype.

- HBM organization: one PIM block per two banks; 16 banks, 8 PIM blocks, 4 banks per bank group, 4 bank groups per pseudo-channel, 4 pseudo-channels per die, and 4 dies per stack. The model supports pseudo-channel mode and treats pseudo-channels as independent.
- Interface: 256-bit prefetch, burst length 4n, 64-bit device/data-bus configuration, modeled 2 Gb/s pin rate, 16 logical channels in `system_hbm.ini`.
- Address mapping: Scheme8, documented as rank → row → column-high → bank-group → bank → channel → column-low → byte offset.
- ISA: 32-bit RISC-style `ADD`, `MUL`, `MAC`, `MAD`, `MOV`, `FILL`, `NOP`, `JUMP`, and `EXIT`; operands include GRF_A, GRF_B, SRF, and bank row buffers. Instructions are programmed into the CRF and memory commands advance execution.
- Data/mode procedure: place data in DRAM; park; SB→HAB; program CRF; HAB→HAB_PIM; execute; HAB_PIM→HAB; HAB→SB; unpark; explicitly read results. GEMV code also uploads input bursts to GRF, uses barriers, and reads accumulated partial sums.
- Built-in kernel mappings: GEMV/GEMV-tree and element-wise ADD, MUL, and ReLU. No native convolution kernel is supplied.
- Validated operand configuration: the locked configuration is FP16, with sixteen FP16 lanes per 256-bit burst. FP32 has an explicit eight-lane arithmetic path but was not validated by this reference run. `INT8` is parsed, but its fallback `BurstType` arithmetic operates on FP16 lanes, so this arm does not claim native validated INT8 execution.
- Absent/unsupported claims: FP8, INT4, block scaling, sparse execution, sparse-index handling, LPDDR5X-PIM, and native convolution.

The exact timing parameters are preserved in the copied `HBM2_samsung_2M_16B_x64.ini`: RL 20, WL 8, tCCDS 2, tCCDL 4, tCCDR 3, tRCDRD 14, tRCDWR 10, tRAS 33, tRRDS 4, tRRDL 6, tRC 47, tRP 14, tRTPS 4, tRTPL 5, tWR 16, tWTRS 4, tWTRL 9, tXAW 16, tRTRS 1, tREFI 3900, tREFISB 121, tRFC 350, tRFCSB 160, tXP 8, tCKE 8, tCMD 1, and tCK 1.

## Downstream phases

The reference gate failure activated the required stop condition:

- Representative ResNet-50 workload preregistration: **NOT RUN**
- Trace/mapping adapter: **NOT IMPLEMENTED**
- Stage 1 cycle measurements and transaction traces: **NOT RUN**
- Analytical additions for transfers, metadata, conversion, or packing: **NOT CREATED**
- Analytical-versus-simulator calibration/validation split: **NOT RUN**
- Joint optimizer integration and exhaustive/sequential/alternating comparisons: **NOT RUN**

No partial test output may be used as a zero-cost placeholder, an unsupported latency estimate, or a Stage 1 optimizer input.

## Review decision

The present unmodified commit/environment combination fails feasibility under the preregistered rule. Any future continuation requires a separately reviewed, post-result preregistration that names the proposed upstream commit/environment or explicit patch and keeps this failed arm intact. It must not tune parameters against a desired joint-gap outcome.
