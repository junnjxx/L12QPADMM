"""Verify the actual MATLAB images against official MNIST train IDX, evaluate on test IDX."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import MNIST
from mat_partitions import Experiment


def load_data(exp: Experiment, root: Path, test_per_class: int, test_seed: int):
    root.mkdir(parents=True, exist_ok=True)
    train = MNIST(root=str(root), train=True, download=True)
    test = MNIST(root=str(root), train=False, download=True)
    original_images = np.asarray(train.data)[exp.train_indices].reshape(1000, 784).astype(np.float32) / 255.0
    original_labels = np.asarray(train.targets, dtype=np.int64)[exp.train_indices]
    if not np.array_equal(exp.labels, original_labels):
        raise ValueError("MAT mnist_labels differ from the ORIGINAL MNIST train labels at mnist_train_indices")
    # Prevent silent synthetic-data use, different MNIST sampling, pixel preprocessing, or index-order mismatch.
    max_diff = float(np.max(np.abs(exp.x - original_images)))
    if max_diff > 2e-6:
        raise ValueError(f"MAT images differ from official MNIST at stored 1-based indices: max_diff={max_diff:.8g}. "
                         "Do NOT train on mismatched data. Check your MATLAB loader and IDX files.")
    images = torch.from_numpy(exp.x.copy()).reshape(-1, 1, 28, 28)
    labels = torch.from_numpy(exp.labels.copy())
    test_images = torch.from_numpy(np.asarray(test.data).copy()).unsqueeze(1).float().div_(255.0)
    test_labels = torch.as_tensor(np.asarray(test.targets).copy(), dtype=torch.long)
    if test_per_class:
        if not 1 <= test_per_class <= 1000:
            raise ValueError("--test-per-class must be between 1 and 1000, or 0 for full 10000")
        rng = np.random.default_rng(test_seed)
        ids = np.concatenate([rng.choice(np.flatnonzero(np.asarray(test.targets) == c),
                                         size=test_per_class, replace=False) for c in range(10)])
        ids = np.sort(ids)
        test_images, test_labels = test_images[ids], test_labels[ids]
    return images, labels, test_images, test_labels, max_diff
