"""MNIST minibatch protocols.

MATLAB, label_balanced and random_fixed: construct one fixed partition and reuse
its ten-image groups throughout training. SHUFFLE_FIXED_BATCH_ORDER=0 preserves
row execution order; =1 reshuffles whole rows each epoch without regrouping.
random_noreplace: redraw a full without-replacement partition every epoch.
"""
from __future__ import annotations

import os

import numpy as np


# Append the new baseline rather than inserting it into the old list. This
# preserves ALL_SCHEMES.index(scheme) (and epoch RNG seeds) for all old methods.
BASELINES = ("random_noreplace", "label_balanced", "random_fixed")
ALL_SCHEMES = (
    "eadmm", "eadmm_2opt", "matrix", "matrix_2opt", "vector", "vector_2opt"
) + BASELINES


def shuffle_fixed_batch_order_enabled() -> bool:
    value = os.environ.get("SHUFFLE_FIXED_BATCH_ORDER", "0").strip()
    if value not in ("0", "1"):
        raise ValueError(
            f"SHUFFLE_FIXED_BATCH_ORDER must be '0' or '1', got {value!r}"
        )
    return value == "1"


def make_fixed_label_balanced_batches(
    labels: np.ndarray,
    rng: np.random.Generator,
    batch_size: int = 10,
) -> np.ndarray:
    """Create one fixed 100x10 partition, one sample per label per row."""
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != 1000 or batch_size != 10:
        raise ValueError(
            "Label-balanced protocol requires 1000 images and batch size 10"
        )
    pools = []
    for cls in range(10):
        members = np.flatnonzero(labels == cls)
        if len(members) != 100:
            raise ValueError(
                f"Label {cls} has {len(members)} samples; expected exactly 100"
            )
        pools.append(rng.permutation(members))
    fixed = np.stack(pools, axis=1).astype(np.int64, copy=False)
    _check_partition(fixed, len(labels), batch_size, "label_balanced")
    return fixed


def make_fixed_random_batches(
    labels: np.ndarray,
    rng: np.random.Generator,
    batch_size: int = 10,
) -> np.ndarray:
    """Draw a 100x10 random partition exactly once per independent seed/trial.

    Do NOT call this function within the epoch loop.
    """
    n = len(labels)
    if n != 1000 or batch_size != 10:
        raise ValueError(
            "Random-fixed protocol requires 1000 images and batch size 10"
        )
    fixed = rng.permutation(n).reshape(-1, batch_size).astype(np.int64)
    _check_partition(fixed, n, batch_size, "random_fixed")
    return fixed


def _check_partition(
    batches: np.ndarray, n: int, batch_size: int, scheme: str
) -> None:
    expected_shape = (n // batch_size, batch_size)
    if batches.shape != expected_shape or not np.array_equal(
        np.sort(batches.ravel()), np.arange(n)
    ):
        raise AssertionError(
            f"Bad epoch minibatch partition for {scheme}: {batches.shape}"
        )


def epoch_batches(
    scheme: str,
    partitions: dict[str, np.ndarray],
    labels: np.ndarray,
    rng: np.random.Generator,
    batch_size: int = 10,
    fixed_balanced_batches: np.ndarray | None = None,
    fixed_random_batches: np.ndarray | None = None,
) -> np.ndarray:
    """Return batches in execution order for one epoch.

    For fixed schemes, only permutations inside each row are generated afresh.
    Set SHUFFLE_FIXED_BATCH_ORDER=1 to additionally change execution order.
    For random_noreplace, redraw all sample-to-batch assignments every epoch.
    """
    labels = np.asarray(labels).reshape(-1)
    n = len(labels)
    if n != 1000 or batch_size != 10:
        raise ValueError(
            "This downstream protocol requires 1000 images and batch size 10"
        )

    if scheme in partitions:
        fixed = np.asarray(partitions[scheme], dtype=np.int64)
    elif scheme == "label_balanced":
        if fixed_balanced_batches is None:
            raise ValueError(
                "label_balanced needs batches created once before the epoch loop"
            )
        fixed = np.asarray(fixed_balanced_batches, dtype=np.int64)
    elif scheme == "random_fixed":
        if fixed_random_batches is None:
            raise ValueError(
                "random_fixed needs batches created once before the epoch loop"
            )
        fixed = np.asarray(fixed_random_batches, dtype=np.int64)
    elif scheme == "random_noreplace":
        fixed = None
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; known: {ALL_SCHEMES}")

    if fixed is None:
        batches = rng.permutation(n).reshape(-1, batch_size)
    else:
        _check_partition(fixed, n, batch_size, scheme)
        # Copy, never mutate MATLAB data or the saved fixed partition.
        batches = np.stack([rng.permutation(row) for row in fixed], axis=0)
        if shuffle_fixed_batch_order_enabled():
            batches = batches[rng.permutation(len(batches)), :]

    _check_partition(batches, n, batch_size, scheme)
    if scheme == "label_balanced":
        target = np.arange(10)
        if any(
            not np.array_equal(np.sort(labels[row]), target)
            for row in batches
        ):
            raise AssertionError("Label-balanced baseline is not balanced")
    return batches
