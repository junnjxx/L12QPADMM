"""Read ONLY the six original minibatch partitions from MATLAB -v7.3 output.

MATLAB stores numeric matrices with reversed HDF5 dimensions. All I arrays
are MATLAB bs x n_batches, 1-based; internally we use n_batches x bs, 0-based.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import hashlib

import h5py
import numpy as np

SCHEMES = ("eadmm", "eadmm_2opt", "matrix", "matrix_2opt", "vector", "vector_2opt")


@dataclass
class Experiment:
    x: np.ndarray  # 1000 x 784, row order is EXACTLY that of MATLAB data
    labels: np.ndarray  # 1000 int64, row-aligned
    train_indices: np.ndarray  # 1000 int64, zero-based original MNIST train IDX indices
    partitions: Dict[str, np.ndarray]  # each: (100, 10), zero-based row indices in x
    mat_sha256: str


def matlab_array(dataset: h5py.Dataset) -> np.ndarray:
    if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "biuf":
        raise ValueError("Expected a real numeric MATLAB v7.3 array; received a nonnumeric HDF5 object")
    a = np.asarray(dataset[()])
    if a.ndim == 2:
        a = a.T
    return a


def require(node: h5py.Group, path: str):
    try:
        return node[path]
    except KeyError as exc:
        raise ValueError(f"MAT file lacks required field: {path}. Do not silently reconstruct a different experiment.") from exc


def integer_array(raw: np.ndarray, label: str) -> np.ndarray:
    a = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
        raise ValueError(f"{label} contains nonfinite or noninteger entries")
    return a.astype(np.int64)


def validate_partition(raw: np.ndarray, name: str, n: int, batch_size: int) -> np.ndarray:
    a = integer_array(raw, name)
    if a.shape != (batch_size, n // batch_size):
        raise ValueError(f"{name}: MATLAB I must be ({batch_size}, {n // batch_size}), got {a.shape}")
    if np.any(a < 1) or np.any(a > n):
        raise ValueError(f"{name} contains MATLAB row indices outside [1, {n}]")
    result = a.T.copy() - 1
    if not np.array_equal(np.sort(result.ravel()), np.arange(n, dtype=np.int64)):
        raise ValueError(f"{name} does not use each of the {n} training samples exactly once")
    return result


def cell_at(f: h5py.File, path: str, matlab_one_based_id: int) -> h5py.Group:
    ds = require(f, path)
    if not isinstance(ds, h5py.Dataset) or ds.dtype.kind != "O":
        raise ValueError(f"{path} must be a MATLAB cell array of HDF5 references")
    # HDF5 stores MATLAB cell dimensions reversed; transpose restores MATLAB order.
    refs = np.asarray(ds[()]).T.flatten(order="F")
    if not 1 <= matlab_one_based_id <= len(refs):
        raise ValueError(f"{path} best start {matlab_one_based_id} is out of range 1..{len(refs)}")
    ref = refs[matlab_one_based_id - 1]
    if not ref:
        raise ValueError(f"{path}[{matlab_one_based_id}] is an empty MATLAB cell")
    group = f[ref]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"{path}[{matlab_one_based_id}] does not contain a result struct")
    return group


def read_experiment(path: str | Path, schemes: tuple[str, ...] = SCHEMES,
                    batch_size: int = 10) -> Experiment:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find MATLAB result MAT: {path}")
    bad = set(schemes) - set(SCHEMES)
    if bad:
        raise ValueError(f"Unknown MATLAB schemes: {sorted(bad)}")
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    with h5py.File(path, "r") as f:
        data = require(f, "data")
        x = np.asarray(matlab_array(require(data, "sample_stationary_points")), dtype=np.float32)
        if x.shape != (1000, 784):
            raise ValueError(f"Expect exactly 1000 MNIST x 784 pixels; got {x.shape}")
        if not np.all(np.isfinite(x)) or x.min() < -1e-6 or x.max() > 1 + 1e-6:
            raise ValueError("MAT data is not normalized [0,1] MNIST pixels. Refusing old synthetic-data experiment")
        labels = integer_array(matlab_array(require(data, "mnist_labels")), "mnist_labels").ravel()
        indices = integer_array(matlab_array(require(data, "mnist_train_indices")), "mnist_train_indices").ravel()
        if len(labels) != 1000 or len(indices) != 1000:
            raise ValueError("MNIST labels/original indices must contain exactly 1000 elements")
        if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 100)):
            raise ValueError("Expect 100 training images per digit, with labels 0..9")
        if indices.min() < 1 or indices.max() > 60000 or len(np.unique(indices)) != 1000:
            raise ValueError("Invalid 1-based training-set IDX indices")
        results = require(f, "results")
        partitions: Dict[str, np.ndarray] = {}
        for name in schemes:
            if name.startswith("eadmm"):
                field = "best_vc2_I" if name.endswith("_2opt") else "best_vc_I"
                raw = matlab_array(require(results, "eadmm/" + field))
            elif name.startswith("vector"):
                field = "I_refined" if name.endswith("_2opt") else "I_raw"
                raw = matlab_array(require(results, "vector/" + field))
            else:
                mid = "best_refined_id" if name.endswith("_2opt") else "best_raw_id"
                best = integer_array(matlab_array(require(results, "matrix/" + mid)), mid).item()
                cell = "refined" if name.endswith("_2opt") else "raw"
                field = "I_after" if name.endswith("_2opt") else "I"
                run = cell_at(f, "results/matrix/" + cell, best)
                raw = matlab_array(require(run, field))
            partitions[name] = validate_partition(raw, name, len(labels), batch_size)
    return Experiment(x, labels, indices - 1, partitions, digest.hexdigest())
