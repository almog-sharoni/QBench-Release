"""Microbenchmark the encoded activation packet round trip on CUDA."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from runspace.src.quantization.activation_transport import (
    encode_dynamic_packet,
    encode_uniform_packet,
)


def _measure(operation, *, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return statistics.median(samples), statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(2026)
    values = torch.randn(args.chunks, 128, device="cuda", dtype=torch.float32)
    same_width_ids = torch.arange(args.chunks, device="cuda", dtype=torch.int64) % 2
    mixed_width_ids = torch.arange(args.chunks, device="cuda", dtype=torch.int64) % 2
    fp8_params = (
        torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        torch.tensor([6, 5], device="cuda", dtype=torch.int32),
        torch.ones(2, device="cuda", dtype=torch.int32),
    )
    mixed_params = (
        torch.tensor([2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([1, 3], device="cuda", dtype=torch.int32),
        torch.ones(2, device="cuda", dtype=torch.int32),
    )

    cases = {
        "uniform_fp8": lambda: encode_uniform_packet(values, "fp8_e4m3").decode(),
        "dynamic_fp8_pair": lambda: encode_dynamic_packet(
            values,
            same_width_ids,
            ("fp8_e1m6", "fp8_e2m5"),
            _candidate_params=fp8_params,
            _trusted_format_ids=True,
        ).decode(),
        "dynamic_mixed_width": lambda: encode_dynamic_packet(
            values,
            mixed_width_ids,
            ("fp4_e2m1", "fp8_e4m3"),
            _candidate_params=mixed_params,
            _trusted_format_ids=True,
        ).decode(),
    }

    print(
        f"device={torch.cuda.get_device_name()} chunks={args.chunks} "
        f"warmup={args.warmup} iterations={args.iterations}"
    )
    for name, operation in cases.items():
        median_ms, mean_ms = _measure(
            operation,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"{name:24s} median={median_ms:9.3f} ms mean={mean_ms:9.3f} ms")


if __name__ == "__main__":
    main()
