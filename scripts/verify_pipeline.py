#!/usr/bin/env python3
"""Verify sample-accurate alignment and the RTNeural JSON export dimensions."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_rtneural_schema(document: dict[str, Any], hidden_size: int) -> None:
    require(document.get("in_shape") == [None, 1], "RTNeural in_shape must be [null, 1]")
    layers = document.get("layers")
    require(isinstance(layers, list) and len(layers) == 2, "Expected exactly LSTM then dense layers")
    lstm, dense = layers
    require(lstm.get("type") == "lstm", "The recurrent layer must use the C++-supported 'lstm' type")
    require(lstm.get("shape") == [None, hidden_size], "LSTM shape does not match hidden size")
    lstm_weights = lstm.get("weights")
    require(isinstance(lstm_weights, list) and len(lstm_weights) == 3, "LSTM needs kernel, recurrent kernel, and bias")
    require(
        [len(lstm_weights[0]), len(lstm_weights[0][0])] == [1, 4 * hidden_size],
        "LSTM kernel must have RTNeural dimensions [input_size, 4 * hidden_size]",
    )
    require(
        [len(lstm_weights[1]), len(lstm_weights[1][0])] == [hidden_size, 4 * hidden_size],
        "LSTM recurrent kernel must have RTNeural dimensions [hidden_size, 4 * hidden_size]",
    )
    require(len(lstm_weights[2]) == 4 * hidden_size, "LSTM bias must contain 4 * hidden_size values")

    require(dense.get("type") == "dense", "The output layer must use the C++-supported 'dense' type")
    require(dense.get("shape") == [None, 1], "Dense output must have one feature")
    require(dense.get("activation") == "", "Dense output activation must be empty/linear")
    dense_weights = dense.get("weights")
    require(isinstance(dense_weights, list) and len(dense_weights) == 2, "Dense layer needs kernel and bias")
    require(
        [len(dense_weights[0]), len(dense_weights[0][0])] == [hidden_size, 1],
        "Dense kernel must have RTNeural dimensions [hidden_size, 1]",
    )
    require(len(dense_weights[1]) == 1, "Dense bias must contain one value")
    require(document.get("metadata", {}).get("sample_rate") == 48000, "Sample-rate metadata is missing")


def verify_alignment(temp_dir: Path) -> None:
    sample_rate = 48000
    rng = np.random.default_rng(42)
    dry = rng.normal(0.0, 0.2, 4096).astype(np.float32)
    dry_path = temp_dir / "dry.wav"

    for expected_lag in (73, -41):
        wet = np.zeros_like(dry)
        if expected_lag > 0:
            wet[expected_lag:] = dry[:-expected_lag]
        else:
            wet[:expected_lag] = dry[-expected_lag:]
        wet_path = temp_dir / f"wet_{expected_lag}.wav"
        output_dir = temp_dir / f"aligned_{expected_lag}"
        sf.write(dry_path, dry, sample_rate, subtype="FLOAT")
        sf.write(wet_path, wet, sample_rate, subtype="FLOAT")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_dataset.py"),
                str(dry_path),
                str(wet_path),
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        require(result.returncode == 0, f"prepare_dataset.py failed: {result.stderr}")
        require(
            f"Estimated lag: {expected_lag} samples" in result.stdout,
            f"Expected lag {expected_lag}, got output: {result.stdout.strip()}",
        )
        aligned_dry, _ = sf.read(output_dir / "train_dry.wav", dtype="float32")
        aligned_wet, _ = sf.read(output_dir / "train_wet.wav", dtype="float32")
        require(
            np.allclose(aligned_dry, aligned_wet, atol=1.0e-6),
            f"Aligned recordings do not match for lag {expected_lag}",
        )
        print(f"Alignment passed: injected lag {expected_lag} samples")


def verify_training_export(temp_dir: Path) -> None:
    rng = np.random.default_rng(7)
    dry = rng.normal(0.0, 0.2, 1024).astype(np.float32)
    wet = np.tanh(2.0 * dry).astype(np.float32)
    train_dry = temp_dir / "train_dry.wav"
    train_wet = temp_dir / "train_wet.wav"
    val_dry = temp_dir / "val_dry.wav"
    val_wet = temp_dir / "val_wet.wav"
    sf.write(train_dry, dry[:768], 48000, subtype="FLOAT")
    sf.write(train_wet, wet[:768], 48000, subtype="FLOAT")
    sf.write(val_dry, dry[768:], 48000, subtype="FLOAT")
    sf.write(val_wet, wet[768:], 48000, subtype="FLOAT")

    models_dir = temp_dir / "models"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "train_lstm.py"),
            "--train-dry",
            str(train_dry),
            "--train-wet",
            str(train_wet),
            "--val-dry",
            str(val_dry),
            "--val-wet",
            str(val_wet),
            "--models-dir",
            str(models_dir),
            "--output-name",
            "verify_model",
            "--hidden-size",
            "5",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--sequence-length",
            "32",
            "--steps-per-epoch",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"train_lstm.py failed: {result.stderr}\n{result.stdout}")

    hidden_size = 5
    model_path = models_dir / "verify_model.json"
    require(model_path.is_file(), "Training did not export its RTNeural JSON model")
    require((models_dir / "verify_model.pt").is_file(), "Training did not save its best checkpoint")
    with model_path.open("r", encoding="utf-8") as model_file:
        document = json.load(model_file)
    assert_rtneural_schema(document, hidden_size)
    require(
        isinstance(document.get("metadata", {}).get("best_val_esr"), float),
        "Training metrics are missing from model metadata",
    )
    print("Training/export passed: RTNeural LSTM and dense dimensions match the C++ loader")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="neural-mvp-verify-") as directory:
        temp_dir = Path(directory)
        verify_alignment(temp_dir)
        verify_training_export(temp_dir)
    print("Pipeline verification passed")


if __name__ == "__main__":
    main()
