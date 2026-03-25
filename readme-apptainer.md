# Apptainer Environment for QBench

This guide provides instructions on how to build and use the Apptainer environment for QBench, which ensures a consistent dependency and CUDA setup.

## Prerequisites
- Apptainer installed on your system.
- NVIDIA drivers and CUDA toolkit available on the host.

## 1. Building the Sandbox

We use a sandbox (a directory-based container) to allow for easy inspection and debugging. To build the sandbox from the definition file:

```bash
apptainer build --fakeroot --sandbox qbench_sandbox Apptainer.def
```

*(Note: If you prefer a single `.sif` file instead of a sandbox, you can run `apptainer build --fakeroot qbench.sif Apptainer.def` instead.)*

## 2. Running QBench Interactive Script

You can run the interactive evaluation script inside the Apptainer environment. We map the host's ImageNet validation dataset into the container using the `--bind` flag, and enable GPU access via the `--nv` flag.

```bash
./run_apptainer.sh
```

This wrapper script automatically handles mapping the ImageNet path and forwarding any arguments (e.g. `--graph-only`) to the interactive script.

### Notes on Overlays
By default, your host's `$PWD` and home directory are automatically mounted into the Apptainer container, allowing the script to read configurations and write outputs directly to your host's filesystem (e.g., inside `runspace/outputs/`). 

Because unprivileged users typically cannot mount `extfs` overlay images (like `.img` files) inside `setuid` Apptainer environments without specific system configurations, we run the container without an overlay flag. This works seamlessly since all required write operations target the naturally mapped host directories.
