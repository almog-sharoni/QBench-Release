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

## Structure

- `src/adapters`: Model adapters (Generic, ResNet, ViT).
- `src/ops`: Quantized operator implementations.
- `src/registry`: Central operator registry.
- `src/eval`: Evaluation logic, metrics, and compliance checkers.
- `src/configs`: YAML configuration files.
- `runspace`: Execution environment for running models and batches.

## Usage

### 1. Clone the Repository
Start by cloning the repository and navigating into it:
```bash
git clone https://github.com/almog-sharoni/QBench-Release.git
cd QBench-Release
```

### 2. Setup Environment (Docker)
The recommended way to run QBench is using the provided Docker container.

1. **Build the Image**:
   *Note: If the `qbench` image already exists on your system, you do not need to rebuild it.*
   ```bash
   docker build -t qbench .
   ```

2. **Start Container**:
   **Important**: Run this command from the root of the cloned repository (where you are now), as it mounts the current directory (`$(pwd)`) to the container.
   ```bash
   docker run -dt --gpus all \
     --shm-size=8g \
     -v "$(pwd)":/app \
     -v /path/to/imagenet:/data/imagenet \
     --name qbench \
     qbench
   ```
   *Note: Replace `/path/to/imagenet` with your actual ImageNet dataset path.*

3. **Enter Container**:
   ```bash
   docker exec -it qbench bash
   ```

### 3. Configure
QBench uses a flexible configuration system.
- **Base Configs**: Located in `runspace/inputs/base_configs/`. Define the quantization parameters (format, mode, etc.).
- **Models**: Located in `runspace/inputs/models.yaml`. Define the models to evaluate.

Example Base Config (`runspace/inputs/base_configs/base_config.yaml`):
```yaml
adapter:
  type: generic
  quantize_first_layer: false
  quantized_ops: ["-1"]

quantization:
  format: fp8_e4m3
  bias: 7
  calib_method: max

dataset:
  name: imagenet
  path: /data/imagenet/val
  batch_size: 64
  num_workers: 4

evaluation:
  mode: compare_only
  compare_batches: 1
```

### 4. Run Evaluation

#### Option A: Interactive Mode (Recommended for single runs)
Use the interactive script to select a configuration and model from a menu.

```bash
python3 runspace/run_interactive.py
```

#### Option B: Batch Mode
Use the `runspace/run_all.py` script to execute a batch of evaluations defined by all base configs and models.

```bash
python3 runspace/run_all.py
```

This script reads the configurations and models defined in `runspace/inputs`, generates specific configurations for each combination, and runs the evaluation.

## Results
Results are organized in the `runspace/outputs` directory.

- **Summary Report**: `runspace/outputs/summary_report.md` (or `.csv`) contains aggregated metrics for all runs.
- **Detailed Reports**: `runspace/outputs/<base_config>/<model>/<format>/` contains:
    - `config.yaml`: The exact configuration used.
    - `comparison_report.txt`: Detailed layer-wise comparison and compliance report.

The summary report includes:
1.  **Accuracy Metrics**: Top-1/Top-5 accuracy and average certainty.
2.  **Compression Stats**: Weight and Input compression reduction percentages.
3.  **Status**: Success/Failure of the run.
