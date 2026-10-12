#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose BatchNorm running statistics in trained MNIST ResNet models.

For each existing checkpoint:

1. Evaluate the original model using its saved BN running statistics.
2. Keep ALL network weights unchanged.
3. Reset BN running statistics.
4. Re-estimate them using only the 1000 training images.
5. Evaluate the same model on the same MNIST test set.
6. Save a comparison CSV.

No optimizer, backward pass, or weight update is performed.

Example:
    python bn_diagnostic.py \
        --run outputs/20260921_150229_3853724
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mat_partitions import read_experiment
from mnist_data import load_data
from models import create_model


HERE = Path(__file__).resolve().parent


def parse_args():

    parser = argparse.ArgumentParser(
        description="BatchNorm recalibration diagnostic."
    )

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Existing experiment output directory containing checkpoints.",
    )

    parser.add_argument(
        "--mnist-root",
        type=Path,
        default=HERE / "data",
    )

    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=100,
        help="Batch size for BN recalibration on the 1000 training images.",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Fixed seed controlling calibration image order.",
    )

    args = parser.parse_args()

    if args.calib_batch_size < 2:
        parser.error("--calib-batch-size must be at least 2")

    if args.test_batch_size < 1:
        parser.error("--test-batch-size must be positive")

    return args


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
):

    # Normal evaluation:
    # BN uses its stored running_mean / running_var.
    model.eval()

    total_loss = 0.0
    total_correct = 0

    for start in range(0, len(labels), batch_size):

        end = min(start + batch_size, len(labels))

        x = images[start:end]
        y = labels[start:end]

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits,
            y,
            reduction="sum",
        )

        total_loss += float(loss.item())

        total_correct += int(
            (logits.argmax(dim=1) == y).sum().item()
        )

    n = len(labels)

    return {
        "loss": total_loss / n,
        "accuracy": total_correct / n,
        "error_pct": 100.0 * (1.0 - total_correct / n),
    }


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    train_images: torch.Tensor,
    batch_size: int,
    seed: int,
):

    """
    Re-estimate BN running statistics using ONLY the training set.

    Important:
        - Network weights are unchanged.
        - No optimizer step.
        - No backward pass.
        - Test images are never used for recalibration.
        - All BN layers use a cumulative average across
          calibration batches (momentum=None).
    """

    bn_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]

    if not bn_layers:
        raise RuntimeError(
            "The loaded model contains no BatchNorm layers."
        )

    # Put the whole model in evaluation mode first.
    # This also disables Dropout if the model has any.
    model.eval()

    original_momentum = {}

    for bn in bn_layers:

        if not bn.track_running_stats:
            raise RuntimeError(
                "BN recalibration requires track_running_stats=True."
            )

        # Preserve the original momentum setting so that
        # the model's configuration remains unchanged afterwards.
        original_momentum[bn] = bn.momentum

        # Reset only running statistics:
        # running_mean, running_var, num_batches_tracked.
        # BN weight and bias are NOT modified.
        bn.reset_running_stats()

        # Cumulative average of calibration-batch statistics.
        bn.momentum = None

        # Enable batch-statistics computation and running-stat updates.
        # Other modules remain in evaluation mode.
        bn.train()

    # All schemes use exactly the same calibration image order.
    rng = np.random.default_rng(seed)

    order = rng.permutation(len(train_images))

    try:

        for start in range(
            0,
            len(train_images),
            batch_size,
        ):

            selected = order[start:start + batch_size]

            idx = torch.as_tensor(
                selected,
                dtype=torch.long,
                device=train_images.device,
            )

            x = train_images[idx]

            # Forward pass only.
            # No labels, no loss, no backward, no weight updates.
            _ = model(x)

    finally:

        # Restore BN momentum and return the model to eval mode.
        for bn in bn_layers:
            bn.momentum = original_momentum[bn]

        model.eval()

    return len(bn_layers)


def load_checkpoint(path: Path, device: torch.device):

    # This script expects checkpoints saved by the
    # existing train.py with --save-checkpoint.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    required = (
        "model_state",
        "model",
        "scheme",
        "seed",
    )

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            f"{path.name}: missing checkpoint fields {missing}"
        )

    model_name = str(checkpoint["model"])

    # The current diagnostic focuses on ResNet16/32.
    if model_name not in ("resnet16", "resnet32"):
        raise ValueError(
            f"Unsupported model in checkpoint: {model_name}"
        )

    model = create_model(model_name)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model = model.to(device)

    return model, checkpoint


def main():

    args = parse_args()

    run_dir = args.run.expanduser().resolve()

    metadata_file = run_dir / "metadata.json"

    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"Cannot find metadata.json: {metadata_file}"
        )

    metadata = json.loads(
        metadata_file.read_text(encoding="utf-8")
    )

    mat_file = Path(metadata["mat_file"])

    if not mat_file.is_file():
        raise FileNotFoundError(
            f"Original MATLAB result file not found: {mat_file}"
        )

    checkpoint_files = sorted(
        run_dir.glob("resnet*_last.pt")
    )

    if not checkpoint_files:

        raise FileNotFoundError(
            "\nNo ResNet checkpoint found in:\n"
            f"  {run_dir}\n\n"
            "The existing training run probably did not use "
            "--save-checkpoint.\n"
            "epochs.csv cannot reconstruct trained network weights."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check nvidia-smi or use --device cpu."
        )

    device = torch.device(args.device)

    set_seed(args.seed)

    print("=" * 70)
    print("BATCHNORM RECALIBRATION DIAGNOSTIC")
    print("=" * 70)

    print(f"[RUN] {run_dir}")
    print(f"[MAT] {mat_file}")
    print(f"[DEVICE] {device}")
    print(f"[CHECKPOINTS] {len(checkpoint_files)}")

    # We only need the saved MNIST images and labels,
    # not MATLAB's six minibatch partitions.
    exp = read_experiment(
        mat_file,
        schemes=(),
    )

    if "mat_sha256" in metadata:

        if exp.mat_sha256 != metadata["mat_sha256"]:
            raise RuntimeError(
                "The current MAT file differs from the one used "
                "in the original training experiment."
            )

    (
        train_x,
        train_y,
        test_x,
        test_y,
        max_diff,
    ) = load_data(
        exp,
        args.mnist_root.expanduser().resolve(),
        int(metadata.get("test_size", 10000)) // 10
        if int(metadata.get("test_size", 10000)) != 10000
        else 0,
        test_seed=123456,
    )

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    test_x = test_x.to(device)
    test_y = test_y.to(device)

    if len(test_y) != int(metadata["test_size"]):
        raise RuntimeError(
            "The test-set size differs from the original experiment."
        )

    print(
        "[DATA] "
        f"train={len(train_y)}, "
        f"test={len(test_y)}, "
        f"max pixel difference={max_diff:.3e}"
    )

    rows = []

    for checkpoint_path in checkpoint_files:

        print()
        print("-" * 70)
        print(f"[CHECKPOINT] {checkpoint_path.name}")

        model, checkpoint = load_checkpoint(
            checkpoint_path,
            device,
        )

        model_name = str(checkpoint["model"])
        scheme = str(checkpoint["scheme"])
        seed = int(checkpoint["seed"])

        # --------------------------------------------------------
        # 1. Original evaluation
        # --------------------------------------------------------

        original = evaluate(
            model,
            test_x,
            test_y,
            args.test_batch_size,
        )

        # --------------------------------------------------------
        # 2. Recalibrate BN on the SAME 1000 training images
        # --------------------------------------------------------

        bn_count = recalibrate_batchnorm(
            model,
            train_x,
            args.calib_batch_size,
            args.seed,
        )

        # --------------------------------------------------------
        # 3. Evaluate again without changing any network weights
        # --------------------------------------------------------

        recalibrated = evaluate(
            model,
            test_x,
            test_y,
            args.test_batch_size,
        )

        row = {
            "model": model_name,
            "scheme": scheme,
            "seed": seed,
            "bn_layers": bn_count,
            "calib_batch_size": args.calib_batch_size,
            "original_test_loss": original["loss"],
            "original_test_error_pct": original["error_pct"],
            "recalibrated_test_loss": recalibrated["loss"],
            "recalibrated_test_error_pct": recalibrated["error_pct"],
            "error_change_pct_points": (
                recalibrated["error_pct"]
                - original["error_pct"]
            ),
        }

        rows.append(row)

        print(f"[MODEL] {model_name}")
        print(f"[SCHEME] {scheme}")
        print(f"[SEED] {seed}")
        print(f"[BN LAYERS] {bn_count}")

        print(
            "[ORIGINAL] "
            f"loss={original['loss']:.6f}, "
            f"test error={original['error_pct']:.4f}%"
        )

        print(
            "[RECALIBRATED] "
            f"loss={recalibrated['loss']:.6f}, "
            f"test error={recalibrated['error_pct']:.4f}%"
        )

        print(
            "[ERROR CHANGE] "
            f"{row['error_change_pct_points']:+.4f} percentage points"
        )

        # Free GPU memory before loading the next checkpoint.
        del model

    # ------------------------------------------------------------
    # 4. Save the diagnostic comparison
    # ------------------------------------------------------------

    output_csv = run_dir / "bn_recalibration.csv"

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print(f"[SAVED] {output_csv}")
    print("[DONE] BN recalibration diagnostic completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()