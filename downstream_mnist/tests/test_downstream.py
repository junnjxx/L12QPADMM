"""Run: python -m unittest discover -s tests -v (no GPU / network required)."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mat_partitions import read_experiment
from batching import ALL_SCHEMES, epoch_batches, make_fixed_label_balanced_batches
from models import create_model


def make_mat(path: Path):
    n, bs, nb = 1000, 10, 100
    standard = (np.arange(n).reshape(nb, bs) + 1).T
    alternate = (np.arange(n).reshape(nb, bs)[:, ::-1] + 1).T
    def put(group, name, value):
        return group.create_dataset(name, data=np.asarray(value, dtype=np.float64).T)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        put(data, "sample_stationary_points", np.zeros((n, 784)))
        put(data, "mnist_labels", np.repeat(np.arange(10), 100).reshape(-1, 1))
        put(data, "mnist_train_indices", np.arange(1, n + 1).reshape(-1, 1))
        results = f.create_group("results")
        e = results.create_group("eadmm")
        put(e, "best_vc_I", standard)
        put(e, "best_vc2_I", alternate)
        v = results.create_group("vector")
        put(v, "I_raw", standard)
        put(v, "I_refined", alternate)
        m = results.create_group("matrix")
        put(m, "best_raw_id", np.array([[2]]))
        put(m, "best_refined_id", np.array([[1]]))
        raw = m.create_dataset("raw", shape=(1, 2), dtype=h5py.ref_dtype)
        refine = m.create_dataset("refined", shape=(1, 2), dtype=h5py.ref_dtype)
        for j in range(2):
            g = f.create_group(f"raw{j}")
            put(g, "I", alternate if j == 1 else standard)
            raw[0, j] = g.ref
            g2 = f.create_group(f"refined{j}")
            put(g2, "I_after", alternate if j == 0 else standard)
            refine[0, j] = g2.ref


class MatlabPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "test.mat"
        make_mat(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_matlab_h5_orientation_cell_ref_best_id_and_2opt(self):
        exp = read_experiment(self.path)
        self.assertEqual(exp.x.shape, (1000, 784))
        self.assertEqual(tuple(exp.partitions), ("eadmm", "eadmm_2opt", "matrix", "matrix_2opt", "vector", "vector_2opt"))
        expected = np.arange(1000).reshape(100, 10)
        self.assertTrue(np.array_equal(exp.partitions["eadmm"], expected))
        self.assertTrue(np.array_equal(exp.partitions["eadmm_2opt"], expected[:, ::-1]))
        self.assertTrue(np.array_equal(exp.partitions["matrix"], expected[:, ::-1]))
        self.assertTrue(np.array_equal(exp.partitions["matrix_2opt"], expected[:, ::-1]))

    def test_all_eight_samplers_each_epoch(self):
        exp = read_experiment(self.path)
        balanced_fixed = make_fixed_label_balanced_batches(exp.labels, np.random.default_rng(2026))
        for scheme in ALL_SCHEMES:
            a = epoch_batches(scheme, exp.partitions, exp.labels, np.random.default_rng(123),
                              fixed_balanced_batches=balanced_fixed)
            b = epoch_batches(scheme, exp.partitions, exp.labels, np.random.default_rng(124),
                              fixed_balanced_batches=balanced_fixed)
            self.assertEqual(a.shape, (100, 10))
            self.assertTrue(np.array_equal(np.sort(a.ravel()), np.arange(1000)))
            self.assertFalse(np.array_equal(a, b))
            if scheme == "label_balanced":
                for row in a:
                    self.assertTrue(np.array_equal(np.sort(exp.labels[row]), np.arange(10)))
                # Every fixed step has exactly the same 10 members in every epoch.
                self.assertTrue(np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)))
                self.assertTrue(np.array_equal(np.sort(a, axis=1), np.sort(balanced_fixed, axis=1)))
            elif scheme in exp.partitions:
                self.assertTrue(np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)))
                self.assertTrue(np.array_equal(np.sort(a, axis=1),
                                               np.sort(exp.partitions[scheme], axis=1)))
            else:
                self.assertFalse(np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1)))

    def test_balanced_cannot_be_redrawn_inside_epoch(self):
        exp = read_experiment(self.path)
        with self.assertRaisesRegex(ValueError, "created ONCE"):
            epoch_batches("label_balanced", exp.partitions, exp.labels, np.random.default_rng(1))

    def test_reject_duplicate_row(self):
        with h5py.File(self.path, "r+") as f:
            d = f["results/vector/I_raw"]
            old = d[0, 0]
            d[0, 0] = d[0, 1]
        try:
            with self.assertRaisesRegex(ValueError, "does not use each"):
                read_experiment(self.path, schemes=("vector",))
        finally:
            with h5py.File(self.path, "r+") as f:
                f["results/vector/I_raw"][0, 0] = old

    def test_reject_old_synthetic_data(self):
        with h5py.File(self.path, "r+") as f:
            d = f["data/sample_stationary_points"]
            old = d[0, 0]
            d[0, 0] = -5
        try:
            with self.assertRaisesRegex(ValueError, "not normalized"):
                read_experiment(self.path, schemes=("eadmm",))
        finally:
            with h5py.File(self.path, "r+") as f:
                f["data/sample_stationary_points"][0, 0] = old

    def test_dry_run_cli(self):
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "train.py"),
                               "--mat", str(self.path), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.count("[OK]"), 8)

    def test_model_output(self):
        import torch
        torch.set_num_threads(1)
        for name in ("resnet16", "resnet32", "mlp"):
            model = create_model(name)
            out = model(torch.zeros(2, 1, 28, 28))
            self.assertEqual(tuple(out.shape), (2, 10))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
