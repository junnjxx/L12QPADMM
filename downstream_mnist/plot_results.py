#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot MNIST downstream results.

Aggregation rule
----------------
WITHIN each experiment directory:
    Same model + scheme, different seeds
    -> calculate their mean at each epoch
    -> plot ONE curve.

ACROSS different experiment directories:
    Never combine seeds or calculate a joint mean.
    Even if model + scheme + seed are identical,
    plot separate curves identified by their source directories.

Examples
--------
Folder A:
    Random, seeds 2026, 2027, 2028
    -> ONE mean curve (n=3).

Folder B:
    Random, seed 2026
    -> ONE separate curve.

Folder A + Folder B:
    -> TWO Random curves, not one combined mean.

Outputs
-------
    train_loss_<model>.png / .pdf
    test_loss_<model>.png / .pdf
    test_error_<model>.png / .pdf

All y-axes use logarithmic scale.

Examples
--------
Single directory:

    python plot_results.py --run outputs/RUN_A

Multiple directories:

    python plot_results.py \
        --runs outputs/RUN_A outputs/RUN_B outputs/RUN_C

Only plot selected schemes:

    python plot_results.py \
        --runs outputs/RUN_A outputs/RUN_B \
        --schemes random_noreplace,random_then_fixed

Specify output directory:

    python plot_results.py \
        --runs outputs/RUN_A outputs/RUN_B \
        --output outputs/my_comparison
"""

from __future__ import annotations

import argparse
import csv
import json

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator


# ============================================================
# 1. Paths and model names
# ============================================================

HERE = Path(__file__).resolve().parent
OUTPUTS_ROOT = HERE / "outputs"

MODEL_ORDER = (
    "mlp",
    "resnet16",
    "resnet32",
    "transformer",
)


# ============================================================
# 2. Scheme order
# ============================================================

SCHEME_ORDER = (
    # Ordinary schemes
    "eadmm",
    "eadmm_2opt",

    "matrix",
    "matrix_2opt",

    "vector",
    "vector_2opt",

    "random_noreplace",
    "random_fixed",
    "label_balanced",

    # Hybrid schemes
    "random_then_eadmm",
    "random_then_eadmm_2opt",

    "random_then_matrix",
    "random_then_matrix_2opt",

    "random_then_vector",
    "random_then_vector_2opt",

    "random_then_label",
    "random_then_fixed",
)


# ============================================================
# 3. Scheme styles
# ============================================================

# Each entry:
#     legend label, color, linestyle

SCHEME_STYLE = {
    # eADMM
    "eadmm": (
        "eADMM",
        "#1f77b4",
        "-",
    ),

    "eadmm_2opt": (
        "eADMM + 2-opt",
        "#1f77b4",
        "--",
    ),

    "random_then_eadmm": (
        "Random -> eADMM",
        "#1f77b4",
        ":",
    ),

    "random_then_eadmm_2opt": (
        "Random -> eADMM + 2-opt",
        "#1f77b4",
        "-.",
    ),

    # Matrix
    "matrix": (
        "Matrix",
        "#d62728",
        "-",
    ),

    "matrix_2opt": (
        "Matrix + 2-opt",
        "#d62728",
        "--",
    ),

    "random_then_matrix": (
        "Random -> Matrix",
        "#d62728",
        ":",
    ),

    "random_then_matrix_2opt": (
        "Random -> Matrix + 2-opt",
        "#d62728",
        "-.",
    ),

    # Vector
    "vector": (
        "Vector",
        "#2ca02c",
        "-",
    ),

    "vector_2opt": (
        "Vector + 2-opt",
        "#2ca02c",
        "--",
    ),

    "random_then_vector": (
        "Random -> Vector",
        "#2ca02c",
        ":",
    ),

    "random_then_vector_2opt": (
        "Random -> Vector + 2-opt",
        "#2ca02c",
        "-.",
    ),

    # Random baselines
    "random_noreplace": (
        "Random (epoch-wise regroup)",
        "#9467bd",
        "-",
    ),

    "random_fixed": (
        "Random-fixed",
        "#8c564b",
        "-",
    ),

    "random_then_fixed": (
        "Random -> Random-fixed",
        "#8c564b",
        "--",
    ),

    # Label-balanced
    "label_balanced": (
        "Label-balanced",
        "#ff7f0e",
        "-",
    ),

    "random_then_label": (
        "Random -> Label-balanced",
        "#ff7f0e",
        "--",
    ),
}


# ============================================================
# 4. Metrics
# ============================================================

METRICS = {
    "train_loss": (
        "Training loss",
        "Training cross-entropy loss",
    ),

    "test_loss": (
        "Test loss",
        "Test cross-entropy loss",
    ),

    "test_error": (
        "Test error",
        "Test error (%)",
    ),
}


REQUIRED_COLUMNS = {
    "scheme",
    "model",
    "seed",
    "epoch",
    "train_loss",
    "test_loss",
    "test_accuracy",
}


# ============================================================
# 5. Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Average independent seeds WITHIN each directory; "
            "keep different directories as separate curves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()

    source.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Two or more experiment directories.",
    )

    source.add_argument(
        "--run",
        type=Path,
        default=None,
        help="One experiment directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Figures are saved in its figures/ subdirectory."
        ),
    )

    parser.add_argument(
        "--schemes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated scheme names. "
            "Default: plot every available scheme."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--fontsize",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    if args.runs is not None and len(args.runs) < 2:
        parser.error(
            "--runs requires at least two directories. "
            "Use --run for a single directory."
        )

    if (
        args.dpi <= 0
        or args.linewidth <= 0
        or args.fontsize <= 0
    ):
        parser.error(
            "--dpi, --linewidth and --fontsize must be positive."
        )

    if args.schemes is not None:

        selected = tuple(
            item.strip().lower()
            for item in args.schemes.split(",")
            if item.strip()
        )

        if not selected:
            parser.error(
                "--schemes must contain at least one scheme."
            )

        if len(selected) != len(set(selected)):
            parser.error(
                "--schemes contains duplicate names."
            )

        args.schemes = selected

    return args


# ============================================================
# 6. Resolve input directories
# ============================================================

def resolve_run_directories(args):

    if args.runs is not None:

        requested = args.runs

    elif args.run is not None:

        requested = [args.run]

    else:

        if not OUTPUTS_ROOT.is_dir():
            raise FileNotFoundError(
                f"Cannot find outputs directory: {OUTPUTS_ROOT}"
            )

        candidates = [
            path
            for path in OUTPUTS_ROOT.iterdir()
            if (
                path.is_dir()
                and (path / "epochs.csv").is_file()
            )
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No epochs.csv found in {OUTPUTS_ROOT}"
            )

        requested = [
            max(
                candidates,
                key=lambda path: path.name,
            )
        ]

    run_dirs = [
        path.expanduser().resolve()
        for path in requested
    ]

    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError(
            "The same directory was supplied more than once."
        )

    for run_dir in run_dirs:

        if not (run_dir / "epochs.csv").is_file():
            raise FileNotFoundError(
                f"Cannot find epochs.csv: {run_dir / 'epochs.csv'}"
            )

    return run_dirs


# ============================================================
# 7. Read CSV records from ONE directory
# ============================================================

def load_records(
    run_dir: Path,
    run_index: int,
):

    csv_path = run_dir / "epochs.csv"

    records = []

    # Deduplication is only performed WITHIN this CSV.
    #
    # Different directories are deliberately kept separate.
    #
    # key:
    #     (model, scheme, seed, epoch)
    #
    # value:
    #     original CSV row and its line number

    seen_within_file = {}

    duplicate_count = 0

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        missing = REQUIRED_COLUMNS - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"{csv_path}: missing columns {sorted(missing)}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):

            if None in row:
                raise ValueError(
                    f"{csv_path}:{line_number}: "
                    "CSV row has more fields than its header."
                )

            try:

                record = {
                    "run_index": run_index,
                    "run_name": run_dir.name,

                    "scheme": row["scheme"].strip().lower(),
                    "model": row["model"].strip().lower(),

                    "seed": int(row["seed"]),
                    "epoch": int(row["epoch"]),

                    "train_loss": float(row["train_loss"]),
                    "test_loss": float(row["test_loss"]),

                    "test_accuracy": float(
                        row["test_accuracy"]
                    ),
                }

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"{csv_path}:{line_number}: malformed row: {exc}"
                ) from exc

            if not record["scheme"] or not record["model"]:
                raise ValueError(
                    f"{csv_path}:{line_number}: "
                    "empty scheme or model."
                )

            if record["epoch"] < 1:
                raise ValueError(
                    f"{csv_path}:{line_number}: "
                    "epoch must be >= 1."
                )

            values = (
                record["train_loss"],
                record["test_loss"],
                record["test_accuracy"],
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{csv_path}:{line_number}: "
                    "non-finite metric."
                )

            if (
                record["train_loss"] < 0
                or record["test_loss"] < 0
            ):
                raise ValueError(
                    f"{csv_path}:{line_number}: negative loss."
                )

            if not (
                0.0 <= record["test_accuracy"] <= 1.0
            ):
                raise ValueError(
                    f"{csv_path}:{line_number}: "
                    "test_accuracy must be between 0 and 1."
                )

            record["test_error"] = (
                1.0 - record["test_accuracy"]
            ) * 100.0

            key = (
                record["model"],
                record["scheme"],
                record["seed"],
                record["epoch"],
            )

            # ------------------------------------------------
            # Identical records repeated WITHIN one CSV:
            #     skip duplicate.
            #
            # Same key but different content WITHIN one CSV:
            #     raise an error.
            #
            # No comparison is made with OTHER directories.
            # ------------------------------------------------

            if key in seen_within_file:

                first_row, first_line = seen_within_file[key]

                if first_row == row:

                    duplicate_count += 1
                    continue

                differing = sorted(
                    column
                    for column in set(first_row) | set(row)
                    if first_row.get(column) != row.get(column)
                )

                raise ValueError(
                    f"Conflicting records inside {csv_path}:\n"
                    f"  key={key}\n"
                    f"  first line={first_line}\n"
                    f"  second line={line_number}\n"
                    f"  differing CSV columns={differing}"
                )

            seen_within_file[key] = (
                dict(row),
                line_number,
            )

            records.append(record)

    if not records:
        raise ValueError(
            f"Empty epochs.csv: {csv_path}"
        )

    if duplicate_count:
        print(
            f"[DEDUP WITHIN FILE] {csv_path}: "
            f"skipped {duplicate_count} identical rows"
        )

    return records


# ============================================================
# 8. Metadata
# ============================================================

def load_metadata(run_dir: Path):

    metadata_file = run_dir / "metadata.json"

    if not metadata_file.is_file():

        print(
            f"[WARNING] metadata.json not found: "
            f"{metadata_file}"
        )

        return {}

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"{metadata_file}: expected a JSON object."
        )

    return metadata


def check_compatibility(
    run_dirs,
    metadatas,
):

    """
    Check that the directories use comparable settings.

    This function ONLY checks metadata.
    It never merges records from different directories.
    """

    if len(run_dirs) < 2:
        return

    comparable_keys = (
        "mat_sha256",

        "normalization",
        "groupnorm_groups",

        "optimizer",
        "learning_rate",
        "weight_decay",

        "train_size",
        "test_size",

        "batch_size",
        "test_per_class",

        "shuffle_fixed_batch_order",
    )

    for key in comparable_keys:

        present = [
            (run_dir.name, metadata[key])
            for run_dir, metadata in zip(
                run_dirs,
                metadatas,
            )
            if key in metadata
        ]

        if len(present) == len(run_dirs):

            first_value = present[0][1]

            if any(
                value != first_value
                for _, value in present[1:]
            ):

                details = "; ".join(
                    f"{name}={value!r}"
                    for name, value in present
                )

                raise ValueError(
                    f"Incompatible experiments for '{key}': "
                    f"{details}"
                )

        elif 0 < len(present) < len(run_dirs):

            print(
                f"[WARNING] Cannot fully compare '{key}': "
                "missing in some metadata files."
            )

    print("[INFO] Training lengths:")

    for run_dir, metadata in zip(
        run_dirs,
        metadatas,
    ):

        print(
            f"       {run_dir.name}: "
            f"{metadata.get('epochs', 'unknown')} epochs"
        )


# ============================================================
# 9. Collect records WITHOUT cross-directory merging
# ============================================================

def collect_records(run_dirs):

    metadatas = [
        load_metadata(run_dir)
        for run_dir in run_dirs
    ]

    check_compatibility(
        run_dirs,
        metadatas,
    )

    all_records = []

    for run_index, run_dir in enumerate(
        run_dirs,
        start=1,
    ):

        records = load_records(
            run_dir,
            run_index,
        )

        all_records.extend(records)

        schemes = sorted(
            set(row["scheme"] for row in records)
        )

        seeds = sorted(
            set(row["seed"] for row in records)
        )

        print(
            f"[INPUT] run={run_index} "
            f"folder={run_dir.name} "
            f"records={len(records)} "
            f"seeds={seeds} "
            f"schemes={schemes}"
        )

    test_sizes = [
        int(metadata["test_size"])
        for metadata in metadatas
        if metadata.get("test_size") is not None
    ]

    if len(test_sizes) == len(run_dirs):

        test_size = test_sizes[0]

    else:

        test_size = None

    if test_size is not None and test_size <= 0:
        raise ValueError(
            "Invalid test_size in metadata.json."
        )

    print(
        f"[COLLECTED] {len(all_records)} records "
        f"from {len(run_dirs)} directories"
    )

    return all_records, test_size


# ============================================================
# 10. Average seeds WITHIN each directory
# ============================================================

def build_series(records):

    """
    ONE curve per:
        (run_index, model, scheme)

    WITHIN that group:
        average across seeds separately for each epoch.

    Different run_index values NEVER share an average.

    Example:
        Run 1: Random seeds 2026, 2027, 2028
            -> 1 mean curve.

        Run 2: Random seed 2026
            -> 1 separate single-seed curve.
    """

    # (run_index, model, scheme)
    #     -> epoch -> seed -> row

    groups = defaultdict(
        lambda: defaultdict(dict)
    )

    for record in records:

        group_key = (
            record["run_index"],
            record["model"],
            record["scheme"],
        )

        epoch = record["epoch"]
        seed = record["seed"]

        if seed in groups[group_key][epoch]:

            raise AssertionError(
                "Duplicate record survived within-file "
                "deduplication: "
                f"group={group_key}, "
                f"seed={seed}, epoch={epoch}"
            )

        groups[group_key][epoch][seed] = record

    series = {}

    for group_key, epoch_data in groups.items():

        run_index, model, scheme = group_key

        epochs = np.asarray(
            sorted(epoch_data),
            dtype=int,
        )

        first_epoch = int(epochs[0])

        first_record = next(
            iter(epoch_data[first_epoch].values())
        )

        all_seeds = sorted(
            {
                seed
                for seed_rows in epoch_data.values()
                for seed in seed_rows
            }
        )

        # Number of seeds contributing to each epoch.
        seed_counts = np.asarray(
            [
                len(epoch_data[int(epoch)])
                for epoch in epochs
            ],
            dtype=int,
        )

        def mean_metric(metric: str) -> np.ndarray:

            return np.asarray(
                [
                    np.mean(
                        [
                            row[metric]
                            for row in epoch_data[
                                int(epoch)
                            ].values()
                        ]
                    )
                    for epoch in epochs
                ],
                dtype=float,
            )

        # ----------------------------------------------------
        # Important:
        #
        # Mean is calculated WITHIN this directory and method.
        #
        # Different directories are never included in the
        # same mean_metric() call.
        # ----------------------------------------------------

        series[group_key] = {
            "run_index": run_index,
            "run_name": first_record["run_name"],

            "model": model,
            "scheme": scheme,

            "seeds": all_seeds,
            "seed_counts": seed_counts,

            "epochs": epochs,

            "train_loss": mean_metric(
                "train_loss"
            ),

            "test_loss": mean_metric(
                "test_loss"
            ),

            "test_error": mean_metric(
                "test_error"
            ),
        }

    return series


# ============================================================
# 11. Curve labels
# ============================================================

def seed_label(item) -> str:

    seeds = item["seeds"]
    counts = item["seed_counts"]

    # One seed in this folder:
    # Show its exact seed ID.

    if len(seeds) == 1:

        return f"seed={seeds[0]}"

    # Multiple seeds with the same number available
    # at every recorded epoch.

    if np.min(counts) == np.max(counts):

        return f"mean, n={int(counts[0])}"

    # Example: some seeds ran 200 epochs;
    # others ran 300 epochs in the SAME folder.

    return (
        f"mean, n={int(np.min(counts))}"
        f"-{int(np.max(counts))} per epoch"
    )


# ============================================================
# 12. Plot one metric for one model
# ============================================================

def plot_one(
    series,
    model,
    metric,
    figure_dir,
    test_size,
    args,
):

    relevant = [
        item
        for item in series.values()
        if item["model"] == model
    ]

    if not relevant:
        return

    available_schemes = {
        item["scheme"]
        for item in relevant
    }

    ordered_schemes = [
        scheme
        for scheme in SCHEME_ORDER
        if scheme in available_schemes
    ]

    ordered_schemes.extend(
        sorted(
            available_schemes - set(SCHEME_ORDER)
        )
    )

    scheme_rank = {
        scheme: index
        for index, scheme in enumerate(
            ordered_schemes
        )
    }

    # Sort by:
    #     method -> source directory

    relevant.sort(
        key=lambda item: (
            scheme_rank[item["scheme"]],
            item["run_index"],
        )
    )

    number_of_curves = len(relevant)

    if number_of_curves >= 15:

        figsize = (12.5, 7.5)

    elif number_of_curves >= 9:

        figsize = (11.5, 6.4)

    else:

        figsize = (11.0, 5.8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": args.fontsize,

            "axes.labelsize": args.fontsize + 1,
            "axes.titlesize": args.fontsize + 2,

            "legend.fontsize": max(
                8,
                args.fontsize - 1,
            ),

            "xtick.labelsize": args.fontsize - 1,
            "ytick.labelsize": args.fontsize - 1,

            "axes.linewidth": 0.9,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    title, ylabel = METRICS[metric]

    all_epochs = []
    zero_found = False

    zero_floor = (
        50.0 / test_size
        if test_size is not None
        else 1e-8
    )

    # Distinguish the SAME scheme appearing in multiple
    # directories without changing its base color.

    folder_styles = (
        "-",
        "--",
        "-.",
        ":",
    )

    # For each scheme, determine its list of source directories.

    scheme_runs = defaultdict(list)

    for item in relevant:

        scheme_runs[
            item["scheme"]
        ].append(
            item["run_index"]
        )

    for item in relevant:

        scheme = item["scheme"]

        epochs = item["epochs"]

        values = item[metric].copy()

        all_epochs.extend(
            epochs.tolist()
        )

        label_base, color, default_linestyle = (
            SCHEME_STYLE.get(
                scheme,
                (
                    scheme,
                    None,
                    "-",
                ),
            )
        )

        # If a scheme exists in only one directory,
        # preserve its original method-specific linestyle.
        #
        # If the SAME scheme appears in multiple directories,
        # assign different linestyles to distinguish them.
        #
        # Legend text always states the source folder.

        source_runs = sorted(
            set(
                scheme_runs[scheme]
            )
        )

        if len(source_runs) == 1:

            linestyle = default_linestyle

        else:

            folder_position = source_runs.index(
                item["run_index"]
            )

            linestyle = folder_styles[
                folder_position % len(folder_styles)
            ]

        if metric == "test_error":

            if np.any(values == 0):
                zero_found = True

            values = np.maximum(
                values,
                zero_floor,
            )

        else:

            values = np.maximum(
                values,
                1e-12,
            )

        # ----------------------------------------------------
        # Source folder is always displayed for multi-folder
        # plots. This prevents a single-seed run from being
        # mistaken for the multi-seed mean from another folder.
        # ----------------------------------------------------

        if args.number_of_runs > 1:

            label = (
                f"{label_base} | "
                f"run {item['run_index']}: "
                f"{item['run_name']} | "
                f"{seed_label(item)}"
            )

        else:

            label = (
                f"{label_base} "
                f"({seed_label(item)})"
            )

        ax.semilogy(
            epochs,
            values,

            label=label,

            color=color,
            linestyle=linestyle,

            linewidth=args.linewidth,
            alpha=0.95,

            solid_capstyle="round",
            dash_capstyle="round",
        )

    if not all_epochs:

        plt.close(fig)
        return

    ax.set_title(
        f"{model.upper()} | {title}",
        pad=12,
        fontweight="semibold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    xmin = min(all_epochs)
    xmax = max(all_epochs)

    ax.set_xlim(
        xmin,
        xmax if xmax > xmin else xmin + 1,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=10,
            integer=True,
        )
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.65,
        alpha=0.25,
    )

    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.45,
        alpha=0.13,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),

        frameon=False,
        ncol=1,

        handlelength=3.0,
        labelspacing=0.65,
    )

    if zero_found:

        ax.text(
            0.0,
            -0.18,

            (
                "Note: zero test errors are displayed at "
                f"{zero_floor:.5g}% only for "
                "logarithmic visualization."
            ),

            transform=ax.transAxes,

            fontsize=max(
                args.fontsize - 3,
                8,
            ),

            color="dimgray",

            ha="left",
            va="top",
        )

    for extension in (
        "png",
        "pdf",
    ):

        destination = (
            figure_dir
            / f"{metric}_{model}.{extension}"
        )

        fig.savefig(
            destination,

            dpi=args.dpi,

            bbox_inches="tight",
            pad_inches=0.18,
        )

        print(
            f"[SAVED] {destination}"
        )

    plt.close(fig)


# ============================================================
# 13. Main
# ============================================================

def main():

    args = parse_args()

    run_dirs = resolve_run_directories(
        args
    )

    args.number_of_runs = len(run_dirs)

    records, test_size = collect_records(
        run_dirs
    )

    # Optional scheme filtering.
    #
    # Filtering does NOT change folder boundaries.
    # Each folder continues to have its own mean.

    if args.schemes is not None:

        available = {
            record["scheme"]
            for record in records
        }

        missing = set(args.schemes) - available

        if missing:

            raise ValueError(
                "Requested schemes not found in supplied CSV files: "
                f"{sorted(missing)}.\n"
                f"Available schemes: {sorted(available)}"
            )

        selected = set(
            args.schemes
        )

        records = [
            record
            for record in records
            if record["scheme"] in selected
        ]

        print(
            "[SELECTED SCHEMES] "
            + ", ".join(args.schemes)
        )

    # This is the ONLY aggregation step.
    #
    # Group key:
    #     (source directory, model, scheme)
    #
    # Means are calculated only across seeds belonging
    # to the SAME source directory.

    series = build_series(
        records
    )

    if args.output is not None:

        base = args.output.expanduser().resolve()

    elif len(run_dirs) == 1:

        base = run_dirs[0]

    else:

        base = OUTPUTS_ROOT / "combined_by_folder"

    figure_dir = (
        base / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[OUTPUT] {figure_dir}"
    )

    print(
        f"[SERIES] {len(series)} curves "
        "across all models"
    )

    print(
        "[AGGREGATION] Mean across seeds within each folder; "
        "no averaging across different folders."
    )

    available_models = {
        item["model"]
        for item in series.values()
    }

    ordered_models = [
        model
        for model in MODEL_ORDER
        if model in available_models
    ]

    ordered_models.extend(
        sorted(
            available_models - set(MODEL_ORDER)
        )
    )

    for model in ordered_models:

        print(
            f"\n[MODEL] {model}"
        )

        for item in sorted(
            (
                item
                for item in series.values()
                if item["model"] == model
            ),
            key=lambda item: (
                item["run_index"],
                item["scheme"],
            ),
        ):

            print(
                f"  [CURVE] run={item['run_index']} "
                f"folder={item['run_name']} "
                f"scheme={item['scheme']} "
                f"seeds={item['seeds']} "
                f"epochs={len(item['epochs'])}"
            )

        for metric in METRICS:

            plot_one(
                series,
                model,
                metric,
                figure_dir,
                test_size,
                args,
            )

    print(
        "\n[DONE] Figures generated from existing CSV files. "
        "Seeds were averaged within each folder only."
    )


if __name__ == "__main__":
    main()