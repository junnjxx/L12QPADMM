#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil

import h5py
import numpy as np


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(
    "/home/ubuntu/xlj/compare_methods/matrix_eadmm_lp_vector/results"
)

OLD = ROOT / (
    "batch_mnist_small1000_N1000_bs10_"
    "Methods_eadmm-random-vector-matrix_"
    "MatrixInit_uniform_EarlyExist1_20260921_111011.mat"
)

NEW = ROOT / (
    "batch_mnist_small1000_N1000_bs10_"
    "Methods_vector_"
    "MatrixInit_uniform_EarlyExist1_20260923_135135.mat"
)

OUTPUT = ROOT / (
    "batch_mnist_small1000_N1000_bs10_"
    "Methods_eadmm-random-vector-matrix_"
    "VectorUpdated_20260923.mat"
)


# ============================================================
# 2. Validate source files
# ============================================================

for path in (OLD, NEW):
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")

    if not h5py.is_hdf5(path):
        raise ValueError(
            f"文件不是 MATLAB v7.3/HDF5 格式：{path}"
        )

if OUTPUT.exists():
    raise FileExistsError(
        f"输出文件已存在，为避免覆盖，请先检查：{OUTPUT}"
    )


# ============================================================
# 3. Confirm that both files use the same problem
# ============================================================

with h5py.File(OLD, "r") as old, h5py.File(NEW, "r") as new:

    fields_to_check = (
        "data/mnist_train_indices",
        "data/mnist_labels",
        "data/sample_stationary_points",
        "problem/Phi",
    )

    for field in fields_to_check:
        if field not in old or field not in new:
            raise KeyError(f"缺少字段：{field}")

        if not np.array_equal(old[field][()], new[field][()]):
            raise ValueError(
                f"新旧 MAT 的 {field} 不一致，停止替换！"
            )

        print(f"[一致] {field}")

    if "results/vector" not in new:
        raise KeyError("新 MAT 中没有 results/vector")

    if "results/vector" not in old:
        raise KeyError("旧 MAT 中没有 results/vector")

    # 确认新结果包含深度学习实验需要的两种分组
    for field in ("I_raw", "I_refined"):
        if field not in new["results/vector"]:
            raise KeyError(
                f"新 MAT 中缺少 results/vector/{field}"
            )

    # 检查每个样本恰好出现一次
    for field in ("I_raw", "I_refined"):
        indices = np.asarray(
            new[f"results/vector/{field}"][()]
        ).ravel()

        if (
            indices.size != 1000
            or not np.array_equal(
                np.sort(indices),
                np.arange(1, 1001),
            )
        ):
            raise ValueError(
                f"新的 Vector {field} 不是有效的完整分组"
            )

        print(f"[有效] 新 Vector {field}")


# ============================================================
# 4. Copy old MAT and replace its Vector results
# ============================================================

shutil.copy2(OLD, OUTPUT)

with h5py.File(NEW, "r") as new, h5py.File(OUTPUT, "r+") as out:

    # 删除复制文件中的旧 Vector 结果
    del out["results/vector"]

    # 将新 Vector 的整个结构复制进去
    new.copy(
        new["results/vector"],
        out["results"],
        name="vector",
    )


# ============================================================
# 5. Verify replacement
# ============================================================

with h5py.File(NEW, "r") as new, h5py.File(OUTPUT, "r") as out:

    for field in ("I_raw", "I_refined"):
        assert np.array_equal(
            new[f"results/vector/{field}"][()],
            out[f"results/vector/{field}"][()],
        )

print()
print("[SUCCESS] Vector 已替换完成！")
print(f"[OUTPUT] {OUTPUT}")
print("[INFO] 旧 MAT 文件未被修改。")