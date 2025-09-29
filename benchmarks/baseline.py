from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import linregress

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curves import fit_gaussian


@dataclass
class Sample:
    x: int
    y: float
    linYError: float = 0.0
    shim: float = 0.0
    scrape: float = 0.0


def samples_recalc(samples: list[Sample]) -> None:
    """Recalculate regression and derived metrics for the provided samples."""
    if len(samples) < 3:
        return

    x_vals = [sample.x for sample in samples]
    y_vals = [sample.y for sample in samples]
    slope, intercept, _, _, _ = linregress(x_vals, y_vals)

    min_y_error = float("inf")
    max_y_error = float("-inf")
    for sample in samples:
        sample.linYError = sample.y - (slope * sample.x + intercept)
        if sample.linYError > max_y_error:
            max_y_error = sample.linYError
        if sample.linYError < min_y_error:
            min_y_error = sample.linYError

    for sample in samples:
        sample.shim = max_y_error - sample.linYError
        sample.scrape = sample.linYError - min_y_error


def _time_block(func: Callable[[], None], iterations: int) -> float:
    """Return total seconds spent running `func` `iterations` times."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return time.perf_counter() - start


def benchmark_fit_gaussian(frames: int, width: int, noise: float, seed: int) -> dict[str, float]:
    """Measure time spent fitting a Gaussian to synthetic histogram data."""
    rng = np.random.default_rng(seed)

    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    base_curve = np.exp(-(x ** 2) / 0.02)  # simple bell curve centred at 0

    def run() -> None:
        curve = base_curve + rng.normal(loc=0.0, scale=noise, size=width)
        fit_gaussian(curve)

    total = _time_block(run, frames)
    return {
        "frames": frames,
        "total_seconds": total,
        "avg_ms": (total / frames) * 1000.0,
        "fps": frames / total if total else math.inf,
    }


def benchmark_samples_recalc(iterations: int, sample_count: int, jitter: float, seed: int) -> dict[str, float]:
    """Measure regression recalculation on a persistent set of samples."""
    rng = np.random.default_rng(seed)

    samples = [Sample(x=index, y=float(rng.normal(loc=0.0, scale=0.5))) for index in range(sample_count)]

    def run() -> None:
        perturb = rng.normal(loc=0.0, scale=jitter, size=sample_count)
        for sample, delta in zip(samples, perturb):
            sample.y += float(delta)
        samples_recalc(samples)

    total = _time_block(run, iterations)
    return {
        "iterations": iterations,
        "total_seconds": total,
        "avg_ms": (total / iterations) * 1000.0,
        "throughput": iterations / total if total else math.inf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark core data-processing paths for the laser-level-webcam project.",
    )
    parser.add_argument("--gaussian-frames", type=int, default=2000, help="Number of synthetic frames to process.")
    parser.add_argument(
        "--frame-width", type=int, default=1920, help="Synthetic histogram width to mimic camera resolution.",
    )
    parser.add_argument(
        "--frame-noise", type=float, default=0.015, help="Gaussian noise factor injected into synthetic frames.",
    )
    parser.add_argument(
        "--samples-iterations", type=int, default=1000, help="How many times to rerun the sample regression.",
    )
    parser.add_argument("--samples-count", type=int, default=180, help="Number of samples in the regression set.")
    parser.add_argument(
        "--samples-jitter", type=float, default=0.002, help="Magnitude of random jitter applied before recalculation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed shared by the synthetic generators.",
    )

    args = parser.parse_args()

    gaussian_stats = benchmark_fit_gaussian(
        frames=args.gaussian_frames,
        width=args.frame_width,
        noise=args.frame_noise,
        seed=args.seed,
    )

    samples_stats = benchmark_samples_recalc(
        iterations=args.samples_iterations,
        sample_count=args.samples_count,
        jitter=args.samples_jitter,
        seed=args.seed + 1,
    )

    print("Gaussian Fit Benchmark")
    print(f"  Frames:       {gaussian_stats['frames']}")
    print(f"  Total time:   {gaussian_stats['total_seconds']:.3f} s")
    print(f"  Avg per run:  {gaussian_stats['avg_ms']:.3f} ms")
    print(f"  Throughput:   {gaussian_stats['fps']:.1f} fps")
    print()
    print("Sample Regression Benchmark")
    print(f"  Iterations:   {samples_stats['iterations']}")
    print(f"  Total time:   {samples_stats['total_seconds']:.3f} s")
    print(f"  Avg per run:  {samples_stats['avg_ms']:.3f} ms")
    print(f"  Throughput:   {samples_stats['throughput']:.1f} ops/s")


if __name__ == "__main__":
    main()

