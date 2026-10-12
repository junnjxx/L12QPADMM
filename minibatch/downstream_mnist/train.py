#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST downstream minibatch comparison.

Ordinary schemes:
    eadmm, eadmm_2opt
    matrix, matrix_2opt
    vector, vector_2opt
    random_noreplace
    label_balanced
    random_fixed

Hybrid schemes:
    random_then_eadmm
    random_then_eadmm_2opt
    random_then_matrix
    random_then_matrix_2opt
    random_then_vector
    random_then_vector_2opt
    random_then_label
    random_then_fixed

Hybrid protocol:
    Epochs 1..switch_epoch:
        random_noreplace.

    Epochs switch_epoch+1..epochs:
        fixed target partition.

Model and optimizer state are preserved across the switch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from batching import (
    ALL_SCHEMES,
    epoch_batches,
    make_fixed_label_balanced_batches,
    make_fixed_random_batches,
)

from mat_partitions import SCHEMES, read_experiment
from mnist_data import load_data
from models import create_model


# ============================================================
# 1. Models and schemes
# ============================================================

MODEL_NAMES = (
    "resnet16",
    "resnet32",
    "mlp",
    "transformer",
)

HYBRID_TARGETS = {
    "random_then_eadmm": "eadmm",
    "random_then_eadmm_2opt": "eadmm_2opt",
    "random_then_matrix": "matrix",
    "random_then_matrix_2opt": "matrix_2opt",
    "random_then_vector": "vector",
    "random_then_vector_2opt": "vector_2opt",
    "random_then_label": "label_balanced",
    "random_then_fixed": "random_fixed",
}

# Do not insert hybrid names into batching.ALL_SCHEMES.
# Its existing indices are used in per-epoch RNG seeds.
RUN_SCHEMES = ALL_SCHEMES + tuple(HYBRID_TARGETS)

STEP_FIELDS = (
    "scheme",
    "model",
    "seed",
    "epoch",
    "step",
    "loss",
    "correct",
    "batch_labels",
    "matlab_row_indices",
    "mnist_train_idx1",
)

EPOCH_FIELDS = (
    "scheme",
    "model",
    "seed",
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "epoch_seconds",
    "elapsed_seconds",
)


# ============================================================
# 2. Arguments
# ============================================================

def parse_csv(value, known, name):
    parts = tuple(
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    )

    if (
        not parts
        or len(set(parts)) != len(parts)
        or set(parts) - set(known)
    ):
        raise argparse.ArgumentTypeError(
            f"{name}: supply unique comma-separated names in {known}"
        )

    return parts


def arguments():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mat", required=True, type=Path)

    p.add_argument(
        "--mnist-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )

    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--schemes", default=",".join(ALL_SCHEMES))

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", default="2026")

    # NEW: Select optimizer without changing training code.
    p.add_argument(
        "--optimizer",
        choices=("sgd", "adam", "adamw"),
        default="sgd",
        help="Optimizer. The previous default remains SGD.",
    )

    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument(
        "--test-per-class",
        type=int,
        default=0,
        help="0: use all 10000 MNIST test images.",
    )

    p.add_argument("--test-batch-size", type=int, default=1024)

    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )

    p.add_argument(
        "--norm",
        choices=("batchnorm", "groupnorm"),
        default="batchnorm",
    )

    p.add_argument("--gn-groups", type=int, default=8)
    p.add_argument("--switch-epoch", type=int, default=50)

    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    args.models = parse_csv(
        args.models,
        MODEL_NAMES,
        "--models",
    )

    args.schemes = parse_csv(
        args.schemes,
        RUN_SCHEMES,
        "--schemes",
    )

    try:
        args.seeds = tuple(
            int(value.strip())
            for value in args.seeds.split(",")
        )
    except ValueError as exc:
        p.error(f"Invalid --seeds: {exc}")

    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
    ):
        p.error("--seeds must contain unique integers")

    if (
        args.epochs <= 0
        or args.test_batch_size <= 0
        or args.learning_rate <= 0
        or args.weight_decay < 0
    ):
        p.error(
            "epochs, test batch size and learning rate must be positive; "
            "weight decay must be nonnegative"
        )

    if args.gn_groups <= 0:
        p.error("--gn-groups must be positive")

    if args.switch_epoch < 0:
        p.error("--switch-epoch must be nonnegative")

    if any(scheme in HYBRID_TARGETS for scheme in args.schemes):
        if not (1 <= args.switch_epoch < args.epochs):
            p.error(
                "Hybrid schemes require "
                "1 <= --switch-epoch < --epochs"
            )

    if args.norm == "groupnorm":
        if any(
            model not in ("resnet16", "resnet32")
            for model in args.models
        ):
            p.error(
                "--norm groupnorm is supported for "
                "ResNet16/ResNet32 only"
            )

    return args


# ============================================================
# 3. Normalization
# ============================================================

def replace_resnet_batchnorm_with_groupnorm(
    model: nn.Module,
    groups: int = 8,
) -> int:
    replaced = 0

    for name, child in model.named_children():

        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features

            if channels % groups != 0:
                raise ValueError(
                    f"--gn-groups={groups} does not divide "
                    f"number of channels={channels}"
                )

            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=channels,
                eps=child.eps,
                affine=child.affine,
            )

            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)

            setattr(model, name, gn)
            replaced += 1

        else:
            replaced += replace_resnet_batchnorm_with_groupnorm(
                child,
                groups,
            )

    return replaced


# ============================================================
# 4. Seeds and optimizer
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
):
    common = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), **common)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **common)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


# ============================================================
# 5. Hybrid scheduling
# ============================================================

def active_scheme_for_epoch(
    run_scheme: str,
    epoch: int,
    switch_epoch: int,
) -> str:
    if run_scheme not in HYBRID_TARGETS:
        return run_scheme

    if epoch <= switch_epoch:
        return "random_noreplace"

    return HYBRID_TARGETS[run_scheme]


# ============================================================
# 6. Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0

    for begin in range(0, len(labels), batch_size):
        end = min(begin + batch_size, len(labels))

        prediction = model(images[begin:end])
        target = labels[begin:end]

        total_loss += float(
            nn.functional.cross_entropy(
                prediction,
                target,
                reduction="sum",
            ).item()
        )

        total_correct += int(
            (
                prediction.argmax(dim=1) == target
            ).sum().item()
        )

    return (
        total_loss / len(labels),
        total_correct / len(labels),
    )


# ============================================================
# 7. Original automatic plot
# ============================================================

def plot_curves(output: Path, records: list[dict]):
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    for model_name in sorted(
        set(row["model"] for row in records)
    ):
        fig, ax = plt.subplots(figsize=(9, 5))

        relevant = [
            row
            for row in records
            if row["model"] == model_name
        ]

        for scheme in sorted(
            set(row["scheme"] for row in relevant)
        ):
            rows = [
                row
                for row in relevant
                if row["scheme"] == scheme
            ]

            epochs = sorted(
                set(row["epoch"] for row in rows)
            )

            mean_accuracy = [
                np.mean(
                    [
                        row["test_accuracy"]
                        for row in rows
                        if row["epoch"] == epoch
                    ]
                )
                for epoch in epochs
            ]

            ax.plot(
                epochs,
                mean_accuracy,
                label=scheme,
            )

        ax.set(
            xlabel="Epoch",
            ylabel="MNIST test accuracy",
            title=(
                f"{model_name}: same 1000 images, "
                "different minibatches"
            ),
            ylim=(0, 1),
        )

        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)

        fig.tight_layout()

        fig.savefig(
            output / f"test_accuracy_{model_name}.png",
            dpi=170,
        )

        plt.close(fig)


# ============================================================
# 8. Main experiment
# ============================================================

def run():
    args = arguments()

    target_schemes = (
        HYBRID_TARGETS.get(scheme, scheme)
        for scheme in args.schemes
    )

    needed_matlab = tuple(
        dict.fromkeys(
            scheme
            for scheme in target_schemes
            if scheme in SCHEMES
        )
    )

    exp = read_experiment(
        args.mat,
        schemes=needed_matlab,
    )

    print(f"[MAT] {args.mat.resolve()}", flush=True)
    print(f"[SHA256] {exp.mat_sha256}", flush=True)

    print(
        f"[DATA] {exp.x.shape}; "
        f"digits={np.bincount(exp.labels, minlength=10).tolist()}; "
        f"MNIST training indices "
        f"{exp.train_indices.min() + 1}.."
        f"{exp.train_indices.max() + 1}",
        flush=True,
    )

    print(
        f"[PARTITIONS] {', '.join(exp.partitions)}",
        flush=True,
    )

    print(
        f"[RUN SCHEMES] {', '.join(args.schemes)}",
        flush=True,
    )

    print(
        f"[CONFIG] optimizer={args.optimizer} "
        f"lr={args.learning_rate} "
        f"weight_decay={args.weight_decay} "
        f"norm={args.norm} "
        f"epochs={args.epochs} "
        f"seeds={args.seeds}",
        flush=True,
    )

    if args.dry_run:
        fixed_balanced = make_fixed_label_balanced_batches(
            exp.labels,
            np.random.default_rng(2026),
        )

        fixed_random = make_fixed_random_batches(
            exp.labels,
            np.random.default_rng(2026),
        )

        for run_scheme in args.schemes:
            if run_scheme in HYBRID_TARGETS:
                active_names = (
                    "random_noreplace",
                    HYBRID_TARGETS[run_scheme],
                )
            else:
                active_names = (run_scheme,)

            for active in active_names:
                batches = epoch_batches(
                    active,
                    exp.partitions,
                    exp.labels,
                    np.random.default_rng(17),
                    fixed_balanced_batches=fixed_balanced,
                    fixed_random_batches=fixed_random,
                )

                print(
                    f"[OK] {run_scheme}, active={active}, "
                    f"shape={batches.shape}",
                    flush=True,
                )

        return

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable. Use --device cpu "
            "or check CUDA_VISIBLE_DEVICES."
        )

    device = torch.device(args.device)

    print(
        f"[DEVICE] {device}: "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    (
        train_x,
        train_y,
        test_x,
        test_y,
        max_diff,
    ) = load_data(
        exp,
        args.mnist_root.expanduser().resolve(),
        args.test_per_class,
        test_seed=123456,
    )

    train_x = train_x.to(device)
    train_y = train_y.to(device)

    test_x = test_x.to(device)
    test_y = test_y.to(device)

    print(
        "[SOURCE VERIFIED] "
        f"MAT vs official MNIST: max pixel diff={max_diff:.3e}; "
        f"train={len(train_y)}; "
        f"test={len(test_y)}",
        flush=True,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = args.output / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{os.getpid()}"
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    has_hybrid = any(
        scheme in HYBRID_TARGETS
        for scheme in args.schemes
    )

    metadata = {
        "mat_file": str(args.mat.resolve()),
        "mat_sha256": exp.mat_sha256,

        "models": args.models,
        "schemes": args.schemes,

        "epochs": args.epochs,
        "seeds": args.seeds,

        "normalization": args.norm,
        "groupnorm_groups": (
            args.gn_groups
            if args.norm == "groupnorm"
            else None
        ),

        # NEW:
        "optimizer": args.optimizer.upper(),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,

        "train_size": len(train_y),
        "test_size": len(test_y),
        "test_per_class": args.test_per_class,

        "batch_size": 10,
        "batches_per_epoch": 100,

        "device": str(device),
        "torch_version": torch.__version__,

        "max_mat_pixel_difference": max_diff,

        "shuffle_fixed_batch_order": os.environ.get(
            "SHUFFLE_FIXED_BATCH_ORDER",
            "0",
        ),

        "random_fixed_partition_seed": (
            "SeedSequence([seed, 990124])"
        ),

        "hybrid_targets": {
            scheme: HYBRID_TARGETS[scheme]
            for scheme in args.schemes
            if scheme in HYBRID_TARGETS
        },

        "switch_epoch": (
            args.switch_epoch
            if has_hybrid
            else None
        ),

        "hybrid_protocol": (
            "Epochs 1..switch_epoch: random_noreplace; "
            "epochs switch_epoch+1..epochs: fixed target. "
            "Model and optimizer state are preserved."
        ),

        "protocol": (
            "Same 1000 MNIST training examples; "
            "MATLAB, label_balanced and random_fixed use "
            "fixed batch membership; random_noreplace "
            "redraws a random partition every epoch."
        ),
    }

    (output / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"[OUTPUT] {output}", flush=True)

    epoch_records: list[dict] = []

    with (
        output / "steps.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as sf, (
        output / "epochs.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as ef:

        sw = csv.DictWriter(sf, STEP_FIELDS)
        ew = csv.DictWriter(ef, EPOCH_FIELDS)

        sw.writeheader()
        ew.writeheader()

        for model_name in args.models:
            for seed in args.seeds:
                for run_scheme in args.schemes:

                    # Same (model, seed) -> same initialization.
                    set_seed(seed)

                    model = create_model(model_name)

                    if args.norm == "groupnorm":
                        count = replace_resnet_batchnorm_with_groupnorm(
                            model,
                            args.gn_groups,
                        )

                        if count == 0:
                            raise RuntimeError(
                                f"No BatchNorm2d layers found "
                                f"in model {model_name}"
                            )

                        print(
                            f"[NORM] {model_name}: "
                            f"replaced {count} BatchNorm2d layers "
                            f"with GroupNorm(groups={args.gn_groups})",
                            flush=True,
                        )

                    model = model.to(device)

                    # NEW: optimizer selection.
                    # Construct only once per run.
                    optimizer = create_optimizer(
                        model=model,
                        optimizer_name=args.optimizer,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                    )

                    target_scheme = HYBRID_TARGETS.get(
                        run_scheme,
                        run_scheme,
                    )

                    fixed_balanced = None

                    if target_scheme == "label_balanced":
                        partition_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, 990123]
                            )
                        )

                        fixed_balanced = make_fixed_label_balanced_batches(
                            exp.labels,
                            partition_rng,
                        )

                    fixed_random = None

                    if target_scheme == "random_fixed":
                        partition_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, 990124]
                            )
                        )

                        fixed_random = make_fixed_random_batches(
                            exp.labels,
                            partition_rng,
                        )

                    method_started = time.perf_counter()

                    print(
                        f"\n[RUN] model={model_name} "
                        f"seed={seed} scheme={run_scheme} "
                        f"optimizer={args.optimizer}",
                        flush=True,
                    )

                    for epoch in range(1, args.epochs + 1):

                        active_scheme = active_scheme_for_epoch(
                            run_scheme,
                            epoch,
                            args.switch_epoch,
                        )

                        if (
                            run_scheme in HYBRID_TARGETS
                            and epoch == args.switch_epoch + 1
                        ):
                            print(
                                f"[SWITCH] {run_scheme}: "
                                f"after epoch {args.switch_epoch}, "
                                f"random_noreplace -> {active_scheme}; "
                                "model/optimizer state preserved",
                                flush=True,
                            )

                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    seed,
                                    epoch,
                                    ALL_SCHEMES.index(active_scheme),
                                ]
                            )
                        )

                        batches = epoch_batches(
                            active_scheme,
                            exp.partitions,
                            exp.labels,
                            rng,
                            fixed_balanced_batches=fixed_balanced,
                            fixed_random_batches=fixed_random,
                        )

                        model.train()

                        epoch_loss = 0.0
                        epoch_correct = 0
                        epoch_started = time.perf_counter()

                        for step, idx in enumerate(
                            batches,
                            start=1,
                        ):
                            ix = torch.as_tensor(
                                idx,
                                device=device,
                                dtype=torch.long,
                            )

                            target = train_y[ix]

                            optimizer.zero_grad(
                                set_to_none=True
                            )

                            prediction = model(train_x[ix])

                            loss = nn.functional.cross_entropy(
                                prediction,
                                target,
                            )

                            loss.backward()
                            optimizer.step()

                            loss_value = float(
                                loss.detach().item()
                            )

                            correct = int(
                                (
                                    prediction.detach().argmax(dim=1)
                                    == target
                                ).sum().item()
                            )

                            epoch_loss += loss_value * len(idx)
                            epoch_correct += correct

                            sw.writerow(
                                dict(
                                    scheme=run_scheme,
                                    model=model_name,
                                    seed=seed,
                                    epoch=epoch,
                                    step=step,
                                    loss=loss_value,
                                    correct=correct,

                                    batch_labels="".join(
                                        str(value)
                                        for value in exp.labels[idx]
                                    ),

                                    matlab_row_indices=" ".join(
                                        str(int(i) + 1)
                                        for i in idx
                                    ),

                                    mnist_train_idx1=" ".join(
                                        str(int(i) + 1)
                                        for i in exp.train_indices[idx]
                                    ),
                                )
                            )

                        test_loss, test_acc = evaluate(
                            model,
                            test_x,
                            test_y,
                            args.test_batch_size,
                        )

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        row = dict(
                            scheme=run_scheme,
                            model=model_name,
                            seed=seed,
                            epoch=epoch,

                            train_loss=epoch_loss / len(train_y),
                            train_accuracy=epoch_correct / len(train_y),

                            test_loss=test_loss,
                            test_accuracy=test_acc,

                            epoch_seconds=(
                                time.perf_counter() - epoch_started
                            ),

                            elapsed_seconds=(
                                time.perf_counter() - method_started
                            ),
                        )

                        epoch_records.append(row)
                        ew.writerow(row)

                        ef.flush()
                        sf.flush()

                        print(
                            f"[EPOCH] "
                            f"{model_name:8s} "
                            f"{run_scheme:26s} "
                            f"optimizer={args.optimizer:5s} "
                            f"active={active_scheme:16s} "
                            f"seed={seed} "
                            f"{epoch:3d}/{args.epochs} "
                            f"train={row['train_accuracy']:.4f} "
                            f"test={test_acc:.4f} "
                            f"loss={row['train_loss']:.5f} "
                            f"time={row['epoch_seconds']:.1f}s",
                            flush=True,
                        )

                    if args.save_checkpoint:
                        checkpoint = {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),

                            "model": model_name,
                            "scheme": run_scheme,
                            "seed": seed,

                            "last_epoch": args.epochs,
                            "mat_sha256": exp.mat_sha256,

                            "optimizer": args.optimizer,
                            "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay,

                            "normalization": args.norm,
                            "groupnorm_groups": (
                                args.gn_groups
                                if args.norm == "groupnorm"
                                else None
                            ),

                            "switch_epoch": (
                                args.switch_epoch
                                if run_scheme in HYBRID_TARGETS
                                else None
                            ),

                            "phase2_scheme": HYBRID_TARGETS.get(
                                run_scheme
                            ),
                        }

                        torch.save(
                            checkpoint,
                            output
                            / (
                                f"{model_name}_{run_scheme}"
                                f"_seed{seed}_last.pt"
                            ),
                        )

                    del optimizer
                    del model

    last = [
        row
        for row in epoch_records
        if row["epoch"] == args.epochs
    ]

    with (output / "final.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, EPOCH_FIELDS)
        writer.writeheader()
        writer.writerows(last)

    plot_curves(output, epoch_records)

    print(
        f"\n[FINISHED] {len(last)} "
        f"model/scheme/seed runs. "
        f"Final results: {output / 'final.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise