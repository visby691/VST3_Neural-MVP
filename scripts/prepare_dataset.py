#!/usr/bin/env python3
"""Align paired dry/wet recordings and split them into training and validation WAVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"{path} must be a mono WAV file (got {audio.ndim} dimensions)")
    if audio.size < 2:
        raise ValueError(f"{path} must contain at least two samples")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path} contains non-finite audio samples")
    return audio, int(sample_rate)


def estimate_lag(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int,
    max_lag_ms: float,
    correlation_window_seconds: float,
) -> int:
    """Return wet's offset relative to dry; positive means wet is delayed."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    window_samples = min(
        len(dry),
        len(wet),
        max(2, int(round(sample_rate * correlation_window_seconds))),
    )
    limit = min(
        int(round(sample_rate * max_lag_ms / 1000.0)),
        window_samples - 1,
    )
    dry_window = dry[:window_samples]
    wet_window = wet[:window_samples]
    dry_centered = dry_window.astype(np.float64, copy=False) - float(np.mean(dry_window))
    wet_centered = wet_window.astype(np.float64, copy=False) - float(np.mean(wet_window))
    if not np.any(dry_centered) or not np.any(wet_centered):
        raise ValueError("Cannot estimate alignment from silent audio")

    correlation = signal.correlate(wet_centered, dry_centered, mode="full", method="fft")
    lags = signal.correlation_lags(len(wet_window), len(dry_window), mode="full")
    allowed = np.abs(lags) <= limit
    if not np.any(allowed):
        raise ValueError("No candidate lags are available; increase --max-lag-ms")
    return int(lags[allowed][np.argmax(correlation[allowed])])


def align_pair(dry: np.ndarray, wet: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        dry = dry[: max(0, len(dry) - lag)]
        wet = wet[lag:]
    elif lag < 0:
        dry = dry[-lag:]
        wet = wet[: max(0, len(wet) + lag)]

    length = min(len(dry), len(wet))
    if length == 0:
        raise ValueError("The estimated lag leaves no overlapping audio")
    return dry[:length], wet[:length]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dry", type=Path, help="Mono DI/dry WAV")
    parser.add_argument("wet", type=Path, help="Mono hardware-processed/wet WAV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--trim-click-seconds",
        type=float,
        default=0.0,
        help="Trim this much from the beginning of both recordings before alignment",
    )
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=1000.0,
        help="Maximum absolute lag searched by cross-correlation (default: 1000 ms)",
    )
    parser.add_argument(
        "--correlation-window-seconds",
        type=float,
        default=10.0,
        help="Leading audio used to estimate lag, limiting memory use (default: 10 s)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of the aligned recording saved for training (default: 0.8)",
    )
    args = parser.parse_args()

    if args.trim_click_seconds < 0:
        parser.error("--trim-click-seconds must not be negative")
    if args.max_lag_ms < 0:
        parser.error("--max-lag-ms must not be negative")
    if args.correlation_window_seconds <= 0:
        parser.error("--correlation-window-seconds must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be between 0 and 1")

    dry, dry_rate = read_mono(args.dry)
    wet, wet_rate = read_mono(args.wet)
    if dry_rate != wet_rate:
        raise ValueError(
            f"Sample-rate mismatch: dry is {dry_rate} Hz, wet is {wet_rate} Hz; "
            "resample both files to the same rate before alignment"
        )

    trim_samples = int(round(args.trim_click_seconds * dry_rate))
    if trim_samples:
        dry = dry[trim_samples:]
        wet = wet[trim_samples:]
    if min(len(dry), len(wet)) < 2:
        raise ValueError("Not enough samples remain after trimming")

    lag = estimate_lag(
        dry, wet, dry_rate, args.max_lag_ms, args.correlation_window_seconds
    )
    print(f"Estimated lag: {lag} samples ({lag / dry_rate * 1000.0:.3f} ms)")
    dry, wet = align_pair(dry, wet, lag)

    split = int(len(dry) * args.train_ratio)
    if split < 1 or split >= len(dry):
        raise ValueError("The aligned audio is too short for a non-empty train/validation split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_dry.wav": dry[:split],
        "train_wet.wav": wet[:split],
        "val_dry.wav": dry[split:],
        "val_wet.wav": wet[split:],
    }
    for filename, audio in outputs.items():
        sf.write(args.output_dir / filename, audio, dry_rate, subtype="FLOAT")
    print(
        f"Saved {split} train and {len(dry) - split} validation samples "
        f"at {dry_rate} Hz to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
