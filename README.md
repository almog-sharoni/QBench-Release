# QBench

A modular and extensible quantization benchmarking framework for researching and evaluating low-precision quantization (FP8, FP4, INT8) on PyTorch models.

## Features

### 1. Multi-Format Quantization
Support for various data formats with simulated quantization:
- **FP8**: E4M3 (Training/Inference), E5M2 (Gradients/Training)
- **FP4**: E2M1, E3M0
- **INT8**: Symmetric quantization

### 2. Flexible Quantization Modes
Configure quantization granularity per-layer or globally:
- **Tensor**: One scale factor per tensor.
- **Channel**: Per-channel scaling (e.g., per output channel for weights, per feature for inputs).
- **Chunk**: Block-wise quantization (e.g., scale factor for every 128 elements), popular in LLMs.

### 3. Granular Configuration
Fine-grained control over how different parts of a layer are quantized:
- **Weights**: `weight_mode`, `weight_chunk_size`
- **Inputs**: `input_mode`, `input_chunk_size`
- **Activations**: `act_mode`, `act_chunk_size` (for ReLU, Softmax, etc.)

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for a detailed reference.

### 4. Comprehensive Layer Support
- **Convolution**: `QuantConv2d` (Weights + Inputs)
- **Linear**: `QuantLinear` (Weights + Inputs)
- **Normalization**: `QuantBatchNorm2d` (Weights + Inputs + Running Stats)
- **Pooling**: `QuantMaxPool2d`, `QuantAdaptiveAvgPool2d` (Inputs)
- **Activations**: `QuantReLU`, `QuantGELU`, `QuantSiLU`, `QuantSoftmax`, etc.

### 5. Advanced Evaluation & Compliance
- **Accuracy**: Top-1/Top-5 Accuracy on ImageNet/Imagenette.
- **Layer-Wise Comparison**: MSE and Cosine Similarity vs. FP32 reference.
- **Compliance Checking**: Verifies that all quantized values (weights, inputs, internal stats) strictly adhere to the target format's representable values.
    - Configurable via `src/eval/compliance_config.yaml`.
    - Generates detailed reports on parameter failures.

## Structure

- `src/adapters`: Model adapters (Generic, ResNet, ViT).
- `src/ops`: Quantized operator implementations.
- `src/registry`: Central operator registry.
- `src/eval`: Evaluation logic, metrics, and compliance checkers.
- `src/configs`: YAML configuration files.
- `tests`: Scripts for evaluation, debugging, and data preparation.

## Usage

### 1. Setup Data
Download and prepare evaluation datasets:
```bash
# For Imagenette (Recommended for quick testing)
python3 tests/download_imagenette.py

# For Tiny ImageNet
python3 tests/download_tiny_imagenet.py
```

### 2. Configure
Edit a configuration file (e.g., `src/configs/fp8_e4m3_chunk128.yaml`) to set your quantization preferences.

```yaml
quantization:
  format: "fp8_e4m3"
  mode: "chunk"
  chunk_size: 128
  weight_mode: "tensor"  # Override for weights
```

### 3. Run Evaluation
```bash
python3 tests/run_eval.py --config src/configs/fp8_e4m3_chunk128.yaml
```

### 4. Run ImageNet Evaluation (Docker)
To run evaluation on the full ImageNet dataset, use the provided Docker container:

1. **Start Container**:
   ```bash
   docker run -dt --gpus all \
     --shm-size=8g \
     -v "$(pwd)":/app \
     -v /data/shared_data/imagenet:/data/imagenet \
     --name qbench \
     qbench
   ```

2. **Run Evaluation**:
   ```bash
   docker exec qbench python3 tests/run_eval.py --config src/configs/imagenet_fp8.yaml
   ```

## Results
Reports are saved to `reports/<model>/<config_name>/comparison_report.txt`.

The report includes:
1.  **Accuracy Metrics**: Top-1/Top-5 accuracy and average certainty.
2.  **FP8 Compliance Check**: Pass/Fail status for weights and parameters.
3.  **Layer-Wise Comparison**: MSE, Cosine Similarity, and Input Dynamic Range.
4.  **Detailed Failures**: Specific examples of non-compliant values if any.
