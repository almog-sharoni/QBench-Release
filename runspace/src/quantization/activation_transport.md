# Activation transport

Layer activation quantization uses producer-output transport boundaries. The
default `encoded` mode packs each selected 128-value chunk into an
`ActivationPacket`; consumers decode that shared packet. `reference` uses the
same FX stage plan but carries the selected decoded FP32 value.

```yaml
evaluation:
  input_quant:
    enabled: true
    mode: dynamic
    transport: encoded
    metric: mse
    chunk_size: 128
    candidate_formats: [fp8_e1m6, fp8_e2m5]
```

`encoded` requires CUDA FP32 activations and `chunk_size: 128`. It is the
default for dynamic, uniform, and model-input-only activation quantization. Use
`transport: reference` explicitly for CPU development or the mathematical
reference path.

Runner also translates the legacy `adapter.input_quantization: true` or
`adapter.output_quantization: true` settings into uniform transport. If the
adapter flag is omitted, the adapter factory's historical input-quantization
default is preserved through encoded transport. Runner always clears both
module fake-quant flags before building the adapter. Set an explicit disabled
`evaluation.input_quant` mapping (or `input_quant: false`) for a weight-only run:

```yaml
evaluation:
  input_quant:
    enabled: false
```

Legacy input and output settings that name different formats are rejected;
one producer packet cannot carry two boundary formats. Use an explicit
`evaluation.input_quant` policy instead. Direct GenericAdapter execution also
fails closed if a non-FP32 module input/output fake-quant flag is live, so the
retired path cannot run silently. Every model returned by a public adapter
`build_model()` call carries the same guard. While transport is installed, and
when activation quantization is explicitly off, legacy internal activation
requantization is disabled too; quantized weight buffers remain independent.

Legacy per-layer `input_format`, `output_format`, `format: fp32`, explicit
output disables, and `skip_quantization` settings are translated into a stage
format policy. Output-only runs do not quantize the external model input.
When producer or fan-out consumer overrides require incompatible formats, the
run fails before inference because one shared packet cannot reproduce both
policies.

The FX planner fuses a compute operation with its single-consumer activation,
so `Conv -> ReLU` is quantized after ReLU rather than before it. ReLU, ReLU6,
Sigmoid, Hardsigmoid, and Softmax producer stages use unsigned formats. In
particular, attention must be decomposed to expose
`score MatMul -> Softmax -> value MatMul`; opaque multi-head attention is
rejected. Runner enables the existing GenericAdapter attention decomposition
automatically for transport-enabled runs.

One packet and format policy is shared across every fan-out edge from a
producer. Dynamic policies that request incompatible candidates for consumers
of the same producer fail with `ActivationProducerPolicyConflict`.

Run results include transport counters and an `activation_plan`. Database runs
persist the same information in `activation_map_json`; the existing
`input_map_json` remains the per-stage format-count map. Both classification
and feature-matching databases retain the activation map, including each
stage's observed packet formats and chunk counts. A versioned semantic
`run_identity` includes the resolved transport and normalized selector metric,
so an old/reference result cannot cause `skip_if_exists` to skip an encoded
hardware run. Execution controls such as `force_rerun` and cache rebuild flags
do not change that identity.

Non-floating tensors are explicit transport bypasses and do not increment
transmission/decode counters. SLM models use a HuggingFace-aware FX trace;
integer token IDs, attention masks, and mask-derived metadata bypass transport,
while floating embedding and decoder activations use the normal stage packets.
Error statistics observe the first consumer decode of each transmission, so
enabling statistics does not add another packet decode and fan-out is counted
once.

The dynamic capture/replay utility records producer-stage format IDs and
replays them through the same transport runtime. INT8 activation packets are
not implemented; entrypoints that request INT8 activation transport reject the
request before evaluation rather than falling back to module fake quantization.
