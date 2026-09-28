#!/usr/bin/env python3
"""Controlled comparison: 6 MATLAB batch partitions + 2 sampling baselines, 3 models."""
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

from batching import ALL_SCHEMES, BASELINES, epoch_batches, make_fixed_label_balanced_batches
from mat_partitions import SCHEMES, read_experiment
from mnist_data import load_data
from models import create_model

MODEL_NAMES = ("resnet16", "resnet32", "mlp", "transformer")
STEP_FIELDS = ("scheme", "model", "seed", "epoch", "step", "loss", "correct", "batch_labels", "matlab_row_indices", "mnist_train_idx1")
EPOCH_FIELDS = ("scheme", "model", "seed", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "epoch_seconds", "elapsed_seconds")


def parse_csv(value: str, known: tuple[str, ...], name: str) -> tuple[str, ...]:
    parts = tuple(i.strip().lower() for i in value.split(",") if i.strip())
    if not parts or len(set(parts)) != len(parts) or set(parts) - set(known):
        raise argparse.ArgumentTypeError(f"{name}: supply unique comma-separated names in {known}")
    return parts


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mat", required=True, type=Path, help="The ONE joint MATLAB result file, saved with -v7.3")
    p.add_argument("--mnist-root", type=Path, default=Path(__file__).resolve().parent / "data", help="Torchvision official MNIST cache")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--schemes", default=",".join(ALL_SCHEMES))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", default="2026", help="Comma-separated independent seeds; use e.g. 2026,2027,2028 for paper")
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--test-per-class", type=int, default=0, help="0: ALL 10000 MNIST test examples; e.g. 100: 1000 test images total")
    p.add_argument("--test-batch-size", type=int, default=1024)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CUDA is required unless explicitly set to cpu")
    p.add_argument("--save-checkpoint", action="store_true", help="Write one LAST-epoch .pt per run; no automatic large checkpoint files")
    p.add_argument("--dry-run", action="store_true", help="Validate MAT schemas and batch partitions only; no training/download")
    args = p.parse_args()
    args.models = parse_csv(args.models, MODEL_NAMES, "--models")
    args.schemes = parse_csv(args.schemes, ALL_SCHEMES, "--schemes")
    try:
        args.seeds = tuple(int(v.strip()) for v in args.seeds.split(","))
    except ValueError as exc:
        p.error(f"Invalid --seeds: {exc}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must contain unique integers")
    if args.epochs <= 0 or args.test_batch_size <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("epochs, test batch size, learning rate must be positive; weight decay nonnegative")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int):
    model.eval()
    total_loss = total_correct = 0.0
    for begin in range(0, len(labels), batch_size):
        end = min(begin + batch_size, len(labels))
        pred = model(images[begin:end])
        target = labels[begin:end]
        total_loss += float(nn.functional.cross_entropy(pred, target, reduction="sum").item())
        total_correct += int((pred.argmax(1) == target).sum().item())
    return total_loss / len(labels), total_correct / len(labels)


def plot_curves(output: Path, records: list[dict]):
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in sorted(set(r["model"] for r in records)):
        fig, ax = plt.subplots(figsize=(9, 5))
        relevant = [r for r in records if r["model"] == name]
        for method in sorted(set(r["scheme"] for r in relevant)):
            rows = [r for r in relevant if r["scheme"] == method]
            epochs = sorted(set(r["epoch"] for r in rows))
            mean = [np.mean([r["test_accuracy"] for r in rows if r["epoch"] == e]) for e in epochs]
            ax.plot(epochs, mean, label=method)
        ax.set(xlabel="Epoch", ylabel="MNIST test accuracy", title=f"{name}: same 1000 images, different minibatches", ylim=(0, 1))
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"test_accuracy_{name}.png", dpi=170)
        plt.close(fig)


def run():
    args = arguments()
    needed_matlab = tuple(s for s in args.schemes if s in SCHEMES)
    exp = read_experiment(args.mat, schemes=needed_matlab)
    print(f"[MAT] {args.mat.resolve()}", flush=True)
    print(f"[SHA256] {exp.mat_sha256}", flush=True)
    print(f"[DATA] {exp.x.shape}; digits {np.bincount(exp.labels, minlength=10).tolist()}; "
          f"MNIST training indices {exp.train_indices.min()+1}..{exp.train_indices.max()+1}", flush=True)
    print(f"[PARTITIONS] {', '.join(exp.partitions)}; baselines: {', '.join(s for s in args.schemes if s in BASELINES)}", flush=True)
    if args.dry_run:
        fixed_balanced = make_fixed_label_balanced_batches(exp.labels, np.random.default_rng(2026))
        for scheme in args.schemes:
            batches = epoch_batches(scheme, exp.partitions, exp.labels, np.random.default_rng(17),
                                    fixed_balanced_batches=fixed_balanced)
            print(f"[OK] {scheme}: {batches.shape}, 1000 unique indices, 0-based range {batches.min()}..{batches.max()}")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run nvidia-smi, verify PyTorch GPU installation, or pass --device cpu explicitly.")
    device = torch.device(args.device)
    print(f"[DEVICE] {device}: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}", flush=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_x, train_y, test_x, test_y, max_diff = load_data(exp, args.mnist_root.expanduser().resolve(),
                                                         args.test_per_class, test_seed=123456)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    print(f"[SOURCE VERIFIED] MAT vs official MNIST: max pixel diff = {max_diff:.3e}; "
          f"train=1000, test={len(test_y)} (official separate test split)", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "mat_file": str(args.mat.resolve()), "mat_sha256": exp.mat_sha256,
        "models": args.models, "schemes": args.schemes, "epochs": args.epochs,
        "seeds": args.seeds, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "train_size": len(train_y), "test_size": len(test_y),
        "batch_size": 10, "batches_per_epoch": 100, "device": str(device),
        "torch_version": torch.__version__, "max_mat_pixel_difference": max_diff,
        "protocol": "same 1000 MATLAB row indices; fixed batch membership AND step order across epochs for MATLAB and label-balanced schemes; only intra-batch shuffle each epoch; random_noreplace redraws per epoch; final epoch reported",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[OUTPUT] {output}", flush=True)
    epoch_records: list[dict] = []
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as sf, \
         (output / "epochs.csv").open("w", newline="", encoding="utf-8") as ef:
        sw, ew = csv.DictWriter(sf, STEP_FIELDS), csv.DictWriter(ef, EPOCH_FIELDS)
        sw.writeheader(); ew.writeheader()
        for model_name in args.models:
            for seed in args.seeds:
                for scheme in args.schemes:
                    # Same (model,seed) starts from exactly the same initialization across ALL schemes.
                    set_seed(seed)
                    model = create_model(model_name).to(device)
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                                  weight_decay=args.weight_decay)
                    # Form the label-balanced partition ONCE for this independent seed.
                    # The seed does not depend on the model, epoch, or run order;
                    # the same (seed, scheme) uses identical batch members in all epochs.
                    fixed_balanced = None
                    if scheme == "label_balanced":
                        partition_rng = np.random.default_rng(
                            np.random.SeedSequence([seed, 990123]))
                        fixed_balanced = make_fixed_label_balanced_batches(exp.labels, partition_rng)
                    method_started = time.perf_counter()
                    print(f"\n[RUN] model={model_name} seed={seed} scheme={scheme}", flush=True)
                    for epoch in range(1, args.epochs + 1):
                        # Use a fresh deterministic generator per (seed,epoch,scheme), without changing the model seed.
                        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, ALL_SCHEMES.index(scheme)]))
                        batches = epoch_batches(scheme, exp.partitions, exp.labels, rng,
                                                fixed_balanced_batches=fixed_balanced)
                        model.train()
                        epoch_loss = 0.0
                        epoch_correct = 0
                        epoch_started = time.perf_counter()
                        for step, idx in enumerate(batches, start=1):
                            ix = torch.as_tensor(idx, device=device, dtype=torch.long)
                            target = train_y[ix]
                            optimizer.zero_grad(set_to_none=True)
                            pred = model(train_x[ix])
                            loss = nn.functional.cross_entropy(pred, target)
                            loss.backward()
                            optimizer.step()
                            loss_value = float(loss.detach().item())
                            correct = int((pred.detach().argmax(1) == target).sum().item())
                            epoch_loss += loss_value * len(idx)
                            epoch_correct += correct
                            sw.writerow(dict(scheme=scheme, model=model_name, seed=seed, epoch=epoch,
                                             step=step, loss=loss_value, correct=correct,
                                             batch_labels="".join(str(x) for x in exp.labels[idx]),
                                             matlab_row_indices=" ".join(str(int(i) + 1) for i in idx),
                                             mnist_train_idx1=" ".join(str(int(i) + 1) for i in exp.train_indices[idx])))
                        test_loss, test_acc = evaluate(model, test_x, test_y, args.test_batch_size)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        row = dict(scheme=scheme, model=model_name, seed=seed, epoch=epoch,
                                   train_loss=epoch_loss / len(train_y), train_accuracy=epoch_correct / len(train_y),
                                   test_loss=test_loss, test_accuracy=test_acc,
                                   epoch_seconds=time.perf_counter() - epoch_started,
                                   elapsed_seconds=time.perf_counter() - method_started)
                        epoch_records.append(row)
                        ew.writerow(row)
                        ef.flush(); sf.flush()
                        print(f"[EPOCH] {model_name:8s} {scheme:16s} seed={seed} {epoch:3d}/{args.epochs} "
                              f"train={row['train_accuracy']:.4f} test={test_acc:.4f} "
                              f"loss={row['train_loss']:.5f} time={row['epoch_seconds']:.1f}s", flush=True)
                    if args.save_checkpoint:
                        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                                    "model": model_name, "scheme": scheme, "seed": seed,
                                    "last_epoch": args.epochs, "mat_sha256": exp.mat_sha256},
                                   output / f"{model_name}_{scheme}_seed{seed}_last.pt")
                    del optimizer, model
    last = [r for r in epoch_records if r["epoch"] == args.epochs]
    with (output / "final.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, EPOCH_FIELDS)
        writer.writeheader(); writer.writerows(last)
    plot_curves(output, epoch_records)
    print(f"\n[FINISHED] {len(last)} model/scheme/seed runs. Final-epoch results: {output / 'final.csv'}", flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
