#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge new vector/vector_2opt training results into an old run.

The output is a PLOT-ONLY directory:
    - Non-vector results come from OLD_RUN.
    - vector and vector_2opt come from NEW_RUN.
    - Original experiment directories are not modified.

Usage:
    python merge_vector_results.py
"""

import csv
import json
import shutil
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

OLD_RUN = ROOT / (
    "outputs/20260923_105517_4107520"
)

NEW_RUN = ROOT / (
    "outputs/adamw_newvector_lr0p001_bn100/"
    "20260923_142230_4127345"
)

MERGED_RUN = ROOT / (
    "outputs/merged_old_methods_new_vector_20260923"
)

REPLACE_SCHEMES = {
    "vector",
    "vector_2opt",
}


# ============================================================
# 2. Helpers
# ============================================================

def load_metadata(run):
    path = run / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparable_value(key, value):
    if key == "optimizer" and value is not None:
        return str(value).lower()

    if key in {"models", "seeds"} and value is not None:
        return tuple(value)

    if key in {"learning_rate", "weight_decay"} and value is not None:
        return float(value)

    return value


def check_compatibility(old_meta, new_meta):
    """
    MAT hashes are intentionally allowed to differ.

    All training configurations that affect the comparison
    must still agree.
    """

    keys = (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "normalization",
        "groupnorm_groups",
        "epochs",
        "seeds",
        "models",
        "train_size",
        "test_size",
        "batch_size",
        "batches_per_epoch",
        "shuffle_fixed_batch_order",
    )

    errors = []

    for key in keys:
        old_value = comparable_value(key, old_meta.get(key))
        new_value = comparable_value(key, new_meta.get(key))

        if old_value != new_value:
            errors.append(
                f"{key}: OLD={old_value!r}; NEW={new_value!r}"
            )

    if errors:
        raise ValueError(
            "两次实验的训练配置不一致，不能直接合并：\n"
            + "\n".join(errors)
            + "\n\n请使用与旧实验完全一致的训练配置重跑 Vector。"
        )


def row_key(row):
    return (
        row["model"],
        row["scheme"],
        int(row["seed"]),
        int(row["epoch"]),
    )


def check_unique(rows, source):
    seen = set()

    for row in rows:
        key = row_key(row)

        if key in seen:
            raise ValueError(
                f"{source} 存在重复记录：{key}"
            )

        seen.add(key)


# ============================================================
# 3. Main
# ============================================================

def main():

    for path in (OLD_RUN, NEW_RUN):
        if not path.is_dir():
            raise FileNotFoundError(
                f"实验目录不存在：{path}"
            )

    if MERGED_RUN.exists():
        raise FileExistsError(
            f"合并目录已经存在，为防止覆盖，请先检查：{MERGED_RUN}"
        )

    old_meta = load_metadata(OLD_RUN)
    new_meta = load_metadata(NEW_RUN)

    print("[OLD MAT]", old_meta.get("mat_file"))
    print("[NEW MAT]", new_meta.get("mat_file"))

    print("[OLD SHA256]", old_meta.get("mat_sha256"))
    print("[NEW SHA256]", new_meta.get("mat_sha256"))

    check_compatibility(old_meta, new_meta)

    print("[OK] 训练配置一致。")

    old_fields, old_rows = read_csv(
        OLD_RUN / "epochs.csv"
    )

    new_fields, new_rows = read_csv(
        NEW_RUN / "epochs.csv"
    )

    if old_fields != new_fields:
        raise ValueError(
            "两份 epochs.csv 的列结构不同，停止合并。"
        )

    check_unique(old_rows, "旧实验")
    check_unique(new_rows, "新实验")

    old_schemes = {row["scheme"] for row in old_rows}
    new_schemes = {row["scheme"] for row in new_rows}

    if not REPLACE_SCHEMES.issubset(old_schemes):
        raise ValueError(
            "旧实验缺少 vector 或 vector_2opt。"
        )

    if not REPLACE_SCHEMES.issubset(new_schemes):
        raise ValueError(
            "新实验缺少 vector 或 vector_2opt。"
        )

    # Check that the new Vector results cover exactly
    # the same model/seed/epoch combinations as before.

    old_vector_keys = {
        row_key(row)
        for row in old_rows
        if row["scheme"] in REPLACE_SCHEMES
    }

    new_vector_rows = [
        row
        for row in new_rows
        if row["scheme"] in REPLACE_SCHEMES
    ]

    new_vector_keys = {
        row_key(row)
        for row in new_vector_rows
    }

    if old_vector_keys != new_vector_keys:
        missing = old_vector_keys - new_vector_keys
        extra = new_vector_keys - old_vector_keys

        raise ValueError(
            "新旧 Vector 的 model/seed/epoch 覆盖范围不一致。\n"
            f"新实验缺少：{len(missing)} 条记录\n"
            f"新实验多出：{len(extra)} 条记录\n"
            "请确认新 Vector 实验已经全部跑完。"
        )

    # Keep all old non-vector schemes.
    retained_rows = [
        row
        for row in old_rows
        if row["scheme"] not in REPLACE_SCHEMES
    ]

    # Insert the newly trained Vector results.
    merged_rows = retained_rows + new_vector_rows

    merged_rows.sort(
        key=lambda row: (
            row["model"],
            int(row["seed"]),
            row["scheme"],
            int(row["epoch"]),
        )
    )

    check_unique(merged_rows, "合并实验")

    # Create an independent plot-only run directory.
    MERGED_RUN.mkdir(parents=True, exist_ok=False)

    write_csv(
        MERGED_RUN / "epochs.csv",
        old_fields,
        merged_rows,
    )

    # Reconstruct final.csv from the last epoch
    # of each model/scheme/seed combination.
    final_epoch = int(old_meta["epochs"])

    final_rows = [
        row
        for row in merged_rows
        if int(row["epoch"]) == final_epoch
    ]

    write_csv(
        MERGED_RUN / "final.csv",
        old_fields,
        final_rows,
    )

    # Preserve the old metadata as the base metadata.
    # The additional fields explicitly identify the
    # different MAT sources used by each method.
    #
    # This is a plot-only merged result, NOT a training run
    # generated entirely from the old MAT file.

    merged_meta = dict(old_meta)

    merged_meta["plot_only_merged_run"] = True

    merged_meta["merge_description"] = (
        "Non-vector schemes: original training run; "
        "vector and vector_2opt: replacement training run."
    )

    merged_meta["merge_sources"] = {
        "original_run": str(OLD_RUN),
        "replacement_run": str(NEW_RUN),
        "replaced_schemes": sorted(REPLACE_SCHEMES),
    }

    merged_meta["mat_sha256_by_scheme"] = {
        scheme: (
            new_meta.get("mat_sha256")
            if scheme in REPLACE_SCHEMES
            else old_meta.get("mat_sha256")
        )
        for scheme in {
            row["scheme"]
            for row in merged_rows
        }
    }

    merged_meta["mat_file_by_scheme"] = {
        scheme: (
            new_meta.get("mat_file")
            if scheme in REPLACE_SCHEMES
            else old_meta.get("mat_file")
        )
        for scheme in {
            row["scheme"]
            for row in merged_rows
        }
    }

    (MERGED_RUN / "metadata.json").write_text(
        json.dumps(
            merged_meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Plotting does not need steps.csv.
    # We intentionally do not copy it because the old
    # Vector step records would be inconsistent with
    # the newly inserted Vector epoch records.

    print()
    print("[SUCCESS] 已生成绘图专用的合并实验目录。")
    print("[OUTPUT]", MERGED_RUN)
    print("[OLD NON-VECTOR ROWS]", len(retained_rows))
    print("[NEW VECTOR ROWS]", len(new_vector_rows))
    print("[TOTAL ROWS]", len(merged_rows))
    print()
    print("原始实验目录均未修改。")


if __name__ == "__main__":
    main()