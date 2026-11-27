#!/usr/bin/env python3
"""Train a sample-by-sample LSTM and export an RTNeural JSON model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import nn


class NeuralLSTM(nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(
        self,
        audio: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features, next_state = self.lstm(audio, state)
        return self.dense(features), next_state


def load_pair(dry_path: Path, wet_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    dry, dry_rate = sf.read(dry_path, dtype="float32", always_2d=False)
    wet, wet_rate = sf.read(wet_path, dtype="float32", always_2d=False)
    if dry.ndim != 1 or wet.ndim != 1:
        raise ValueError(f"Training inputs must be mono: {dry_path} and {wet_path}")
    if dry_rate != wet_rate:
        raise ValueError(f"Sample-rate mismatch: {dry_path} is {dry_rate} Hz, {wet_path} is {wet_rate} Hz")
    if len(dry) != len(wet):
        raise ValueError(f"Dry/wet lengths differ: {dry_path} has {len(dry)}, {wet_path} has {len(wet)}")
    if len(dry) < 2:
        raise ValueError(f"At least two paired samples are required: {dry_path}, {wet_path}")
    if not np.isfinite(dry).all() or not np.isfinite(wet).all():
        raise ValueError(f"Non-finite samples found in {dry_path} or {wet_path}")
    return dry, wet, int(dry_rate)


def preemphasis(values: torch.Tensor, alpha: float) -> torch.Tensor:
    previous = torch.cat((torch.zeros_like(values[:, :1]), values[:, :-1]), dim=1)
    return values - alpha * previous


def batch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    preemphasis_coefficient: float,
    dc_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = prediction[..., 0]
    reference = target[..., 0]
    pred_filtered = preemphasis(pred, preemphasis_coefficient)
    target_filtered = preemphasis(reference, preemphasis_coefficient)
    error_energy = torch.sum((pred_filtered - target_filtered) ** 2)
    signal_energy = torch.sum(target_filtered**2)
    esr = error_energy / (signal_energy + 1.0e-12)
    dc_loss = torch.mean((torch.mean(pred, dim=1) - torch.mean(reference, dim=1)) ** 2)
    return esr + dc_loss_weight * dc_loss, esr


def evaluate_esr(
    model: NeuralLSTM,
    dry: np.ndarray,
    wet: np.ndarray,
    device: torch.device,
    sequence_length: int,
    alpha: float,
) -> float:
    model.eval()
    hidden_state = None
    previous_prediction = 0.0
    previous_target = 0.0
    error_energy = 0.0
    signal_energy = 0.0

    with torch.no_grad():
        for start in range(0, len(dry), sequence_length):
            end = min(start + sequence_length, len(dry))
            inputs = torch.from_numpy(dry[start:end]).to(device=device).view(1, -1, 1)
            targets = torch.from_numpy(wet[start:end]).to(device=device).view(1, -1, 1)
            predictions, hidden_state = model(inputs, hidden_state)

            pred = predictions[0, :, 0]
            target = targets[0, :, 0]
            pred_previous = torch.cat(
                (pred.new_tensor([previous_prediction]), pred[:-1])
            )
            target_previous = torch.cat(
                (target.new_tensor([previous_target]), target[:-1])
            )
            pred_filtered = pred - alpha * pred_previous
            target_filtered = target - alpha * target_previous
            error_energy += float(torch.sum((pred_filtered - target_filtered) ** 2).item())
            signal_energy += float(torch.sum(target_filtered**2).item())
            previous_prediction = float(pred[-1].item())
            previous_target = float(target[-1].item())

    return error_energy / (signal_energy + 1.0e-12)


def _as_finite_list(tensor: torch.Tensor) -> list[Any]:
    values = tensor.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Cannot export model weights containing non-finite values")
    return values.tolist()


def export_rtneural_json(
    model: NeuralLSTM,
    output_path: Path,
    sample_rate: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Export the RTNeural dynamic parser's in_shape/layers/weights schema."""
    model.eval()
    hidden_size = model.lstm.hidden_size
    input_weights = model.lstm.weight_ih_l0.detach().T
    recurrent_weights = model.lstm.weight_hh_l0.detach().T
    # RTNeural expects a single bias vector; PyTorch stores input and recurrent biases separately.
    bias = model.lstm.bias_ih_l0.detach() + model.lstm.bias_hh_l0.detach()
    layers = [
        {
            "type": "lstm",
            "shape": [None, hidden_size],
            "weights": [
                _as_finite_list(input_weights),
                _as_finite_list(recurrent_weights),
                _as_finite_list(bias),
            ],
        },
        {
            "type": "dense",
            "activation": "",
            "shape": [None, 1],
            "weights": [
                _as_finite_list(model.dense.weight.detach().T),
                _as_finite_list(model.dense.bias.detach()),
            ],
        },
    ]
    document = {
        "in_shape": [None, 1],
        "layers": layers,
        "metadata": {
            **(metadata or {}),
            "sample_rate": int(sample_rate),
            "architecture": "RTNeural LSTM + dense",
            "hidden_size": hidden_size,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dry", type=Path, default=Path("data/processed/train_dry.wav"))
    parser.add_argument("--train-wet", type=Path, default=Path("data/processed/train_wet.wav"))
    parser.add_argument("--val-dry", type=Path, default=Path("data/processed/val_dry.wav"))
    parser.add_argument("--val-wet", type=Path, default=Path("data/processed/val_wet.wav"))
    parser.add_argument("--models-dir", type=Path, default=repo_root / "models")
    parser.add_argument("--output-name", default="neural_lstm")
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--preemphasis", type=float, default=0.95)
    parser.add_argument("--dc-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hidden_size < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("--hidden-size, --epochs, and --batch-size must be positive")
    if args.sequence_length < 1 or args.steps_per_epoch < 1 or args.learning_rate <= 0:
        parser.error("--sequence-length and --steps-per-epoch must be positive; learning rate must be positive")
    if not 0.0 <= args.preemphasis < 1.0 or args.dc_loss_weight < 0.0:
        parser.error("--preemphasis must be in [0, 1), and --dc-loss-weight must not be negative")

    train_dry, train_wet, sample_rate = load_pair(args.train_dry, args.train_wet)
    val_dry, val_wet, val_rate = load_pair(args.val_dry, args.val_wet)
    if sample_rate != val_rate:
        raise ValueError(f"Train/validation sample rates differ: {sample_rate} Hz vs {val_rate} Hz")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; sample rate: {sample_rate} Hz")

    model = NeuralLSTM(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_esr = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    steps = args.steps_per_epoch

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_error = 0.0
        for _ in range(steps):
            starts = np.random.randint(
                0, max(1, len(train_dry) - args.sequence_length + 1), size=args.batch_size
            )
            dry_batch = np.stack(
                [train_dry[start : start + args.sequence_length] for start in starts]
            )
            wet_batch = np.stack(
                [train_wet[start : start + args.sequence_length] for start in starts]
            )
            inputs = torch.from_numpy(dry_batch).to(device=device).unsqueeze(-1)
            targets = torch.from_numpy(wet_batch).to(device=device).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(inputs)
            loss, esr = batch_loss(prediction, targets, args.preemphasis, args.dc_loss_weight)
            loss.backward()
            optimizer.step()

            train_error += float(esr.detach().item())

        train_esr = train_error / steps
        val_esr = evaluate_esr(
            model, val_dry, val_wet, device, args.sequence_length, args.preemphasis
        )
        print(f"Epoch {epoch:03d}/{args.epochs}: Train ESR {train_esr:.6f} | Val ESR {val_esr:.6f}")

        if val_esr < best_val_esr:
            best_val_esr = val_esr
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training completed without a finite validation result")
    model.load_state_dict(best_state)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{args.output_name}.pt"
    json_path = args.models_dir / f"{args.output_name}.json"
    torch.save(
        {
            "state_dict": best_state,
            "hidden_size": args.hidden_size,
            "sample_rate": sample_rate,
            "best_val_esr": best_val_esr,
        },
        checkpoint_path,
    )
    export_rtneural_json(
        model,
        json_path,
        sample_rate,
        {
            "best_val_esr": best_val_esr,
            "preemphasis": args.preemphasis,
            "dc_loss_weight": args.dc_loss_weight,
        },
    )
    print(f"Best Val ESR: {best_val_esr:.6f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Exported RTNeural model: {json_path}")


if __name__ == "__main__":
    main()
