#!/usr/bin/env python
"""
Generate the locked BOSSBase 0.4 bpp Q10-Q90 detector-signature
interval heatmap used by the HSNNET ranked-attribution package.

Each matrix cell represents one target-algorithm/detector-dimension
pair. The cell annotation reports the exact [Q10, Q90] interval,
while the background color represents the interval midpoint.

Diagonal boxes identify matched algorithm-detector pairs.
"""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ALGORITHM_ORDER = [
    "WOW",
    "S-UNIWARD",
    "HILL",
    "HUGO",
    "MiPOD",
]

EXPECTED_COLUMNS = {
    "target_algo",
    "template_dimension",
    "q_low_used",
    "q_high_used",
    "q10",
    "q90",
}

EXPECTED_DIAGONAL = {
    "WOW": {
        "q10": 1.405922,
        "q90": 3.558374,
    },
    "S-UNIWARD": {
        "q10": 0.887013,
        "q90": 3.067368,
    },
    "HILL": {
        "q10": 0.865907,
        "q90": 2.819321,
    },
    "HUGO": {
        "q10": 1.082785,
        "q90": 2.974571,
    },
    "MiPOD": {
        "q10": 0.793971,
        "q90": 2.564312,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def normalize_algorithm(value: object) -> str:
    text = str(value).strip()
    key = text.upper().replace("-", "_").replace(" ", "_")

    mapping = {
        "WOW": "WOW",
        "S_UNIWARD": "S-UNIWARD",
        "HILL": "HILL",
        "HUGO": "HUGO",
        "MIPOD": "MiPOD",
    }

    return mapping.get(key, text)


def validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = EXPECTED_COLUMNS - set(frame.columns)

    if missing_columns:
        raise RuntimeError(
            "Signature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    work = frame.copy()

    work["target_algo"] = (
        work["target_algo"]
        .map(normalize_algorithm)
    )

    work["template_dimension"] = (
        work["template_dimension"]
        .map(normalize_algorithm)
    )

    for column in (
        "q_low_used",
        "q_high_used",
        "q10",
        "q90",
    ):
        work[column] = pd.to_numeric(
            work[column],
            errors="raise",
        )

    if len(work) != 25:
        raise RuntimeError(
            f"Expected 25 rows, found {len(work)}."
        )

    if set(work["target_algo"]) != set(ALGORITHM_ORDER):
        raise RuntimeError(
            "Unexpected target-algorithm set."
        )

    if set(work["template_dimension"]) != set(ALGORITHM_ORDER):
        raise RuntimeError(
            "Unexpected detector-dimension set."
        )

    pair_columns = [
        "target_algo",
        "template_dimension",
    ]

    if work[pair_columns].duplicated().any():
        raise RuntimeError(
            "Duplicate target-detector pairs were found."
        )

    if work[pair_columns].drop_duplicates().shape[0] != 25:
        raise RuntimeError(
            "The target-detector grid is incomplete."
        )

    if not (work["q_low_used"] == 10).all():
        raise RuntimeError(
            "q_low_used is not locked to 10."
        )

    if not (work["q_high_used"] == 90).all():
        raise RuntimeError(
            "q_high_used is not locked to 90."
        )

    if (work["q10"] > work["q90"]).any():
        raise RuntimeError(
            "At least one interval has Q10 greater than Q90."
        )

    for algorithm, expected in EXPECTED_DIAGONAL.items():
        row = work[
            (work["target_algo"] == algorithm)
            & (
                work["template_dimension"]
                == algorithm
            )
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Missing diagonal row for {algorithm}."
            )

        actual_q10 = float(row.iloc[0]["q10"])
        actual_q90 = float(row.iloc[0]["q90"])

        if not np.isclose(
            actual_q10,
            expected["q10"],
            atol=5e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Unexpected diagonal Q10 for {algorithm}: "
                f"{actual_q10}"
            )

        if not np.isclose(
            actual_q90,
            expected["q90"],
            atol=5e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Unexpected diagonal Q90 for {algorithm}: "
                f"{actual_q90}"
            )

    return work


def build_matrix(
    work: pd.DataFrame,
    value_column: str,
) -> np.ndarray:
    pivot = work.pivot(
        index="target_algo",
        columns="template_dimension",
        values=value_column,
    )

    pivot = pivot.reindex(
        index=ALGORITHM_ORDER,
        columns=ALGORITHM_ORDER,
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            f"Incomplete matrix for {value_column}."
        )

    return pivot.to_numpy(dtype=float)


def generate_figure(
    q10: np.ndarray,
    q90: np.ndarray,
    png_path: Path,
    pdf_path: Path,
) -> None:
    midpoint = (q10 + q90) / 2.0

    maximum_absolute_midpoint = float(
        np.max(np.abs(midpoint))
    )

    if maximum_absolute_midpoint == 0:
        maximum_absolute_midpoint = 1.0

    norm = TwoSlopeNorm(
        vmin=-maximum_absolute_midpoint,
        vcenter=0.0,
        vmax=maximum_absolute_midpoint,
    )

    figure, axis = plt.subplots(
        figsize=(11.8, 8.1),
        constrained_layout=False,
    )

    image = axis.imshow(
        midpoint,
        cmap="coolwarm",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    axis.set_xticks(
        np.arange(len(ALGORITHM_ORDER))
    )

    axis.set_yticks(
        np.arange(len(ALGORITHM_ORDER))
    )

    axis.set_xticklabels(
        ALGORITHM_ORDER,
        fontsize=11,
    )

    axis.set_yticklabels(
        ALGORITHM_ORDER,
        fontsize=11,
    )

    axis.set_xlabel(
        "Detector dimension",
        fontsize=12,
        labelpad=10,
    )

    axis.set_ylabel(
        "Candidate algorithm",
        fontsize=12,
        labelpad=10,
    )

    axis.set_title(
        "Validation response signatures (Q10–Q90)\n"
        "BOSSBase 0.4 bpp",
        fontsize=15,
        pad=17,
    )

    # Grid boundaries.
    axis.set_xticks(
        np.arange(-0.5, len(ALGORITHM_ORDER), 1),
        minor=True,
    )

    axis.set_yticks(
        np.arange(-0.5, len(ALGORITHM_ORDER), 1),
        minor=True,
    )

    axis.grid(
        which="minor",
        linewidth=1.1,
        alpha=0.85,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for row_index in range(len(ALGORITHM_ORDER)):
        for column_index in range(len(ALGORITHM_ORDER)):
            low = q10[row_index, column_index]
            high = q90[row_index, column_index]
            normalized_value = norm(
                midpoint[row_index, column_index]
            )

            text_color = (
                "white"
                if (
                    normalized_value < 0.23
                    or normalized_value > 0.77
                )
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"[{low:.3f},\n{high:.3f}]",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight=(
                    "bold"
                    if row_index == column_index
                    else "normal"
                ),
                color=text_color,
            )

            if row_index == column_index:
                axis.add_patch(
                    Rectangle(
                        (
                            column_index - 0.46,
                            row_index - 0.46,
                        ),
                        0.92,
                        0.92,
                        fill=False,
                        linewidth=2.8,
                        edgecolor="black",
                    )
                )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Interval midpoint (standardized response)",
        fontsize=11,
    )

    figure.text(
        0.5,
        0.025,
        "Cell text reports the exact [Q10, Q90] interval; "
        "boxed diagonal cells are matched algorithm-detector pairs.",
        ha="center",
        va="center",
        fontsize=9.5,
    )

    figure.subplots_adjust(
        left=0.16,
        right=0.89,
        top=0.84,
        bottom=0.13,
    )

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": (
                "HSNNET Q10-Q90 validation response "
                "signature heatmap"
            ),
            "Description": (
                "BOSSBase 0.4 bpp validation signature "
                "intervals over five candidate algorithms "
                "and five detector dimensions."
            ),
        },
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": (
                "HSNNET Q10-Q90 validation response "
                "signature heatmap"
            ),
            "Author": "HSNNET reproducibility package",
            "Subject": (
                "BOSSBase 0.4 bpp validation signature "
                "intervals"
            ),
            "Keywords": (
                "HSNNET, steganalysis, Q10, Q90, "
                "detector bank, attribution"
            ),
            "CreationDate": None,
            "ModDate": None,
        },
    )

    plt.close(figure)


def main() -> None:
    repository_root = (
        Path(__file__).resolve().parents[1]
    )

    source_path = (
        repository_root
        / "results"
        / "core"
        / "bossbase_04"
        / "final_cross_template_signature_table.csv"
    )

    output_directory = (
        repository_root
        / "figures"
    )

    png_path = (
        output_directory
        / "q10_q90_signature_interval_heatmap.png"
    )

    pdf_path = (
        output_directory
        / "q10_q90_signature_interval_heatmap.pdf"
    )

    metadata_path = (
        output_directory
        / "q10_q90_signature_interval_heatmap_metadata.json"
    )

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_frame = pd.read_csv(source_path)
    work = validate_source(source_frame)

    q10 = build_matrix(work, "q10")
    q90 = build_matrix(work, "q90")

    generate_figure(
        q10=q10,
        q90=q90,
        png_path=png_path,
        pdf_path=pdf_path,
    )

    metadata = {
        "figure_name": (
            "Q10-Q90 validation response "
            "signature interval heatmap"
        ),
        "dataset": "BOSSBase 1.01",
        "payload_bpp": 0.4,
        "source_file": str(
            source_path.relative_to(repository_root)
        ).replace("\\", "/"),
        "source_sha256": sha256_file(source_path),
        "target_axis_column": "target_algo",
        "detector_axis_column": "template_dimension",
        "lower_bound_column": "q10",
        "upper_bound_column": "q90",
        "q_low_used": 10,
        "q_high_used": 90,
        "algorithm_order": ALGORITHM_ORDER,
        "matrix_shape": [5, 5],
        "validated_pairs": 25,
        "color_encoding": (
            "Arithmetic midpoint of Q10 and Q90"
        ),
        "cell_annotation": (
            "Exact [Q10, Q90] interval, three decimals"
        ),
        "diagonal_encoding": (
            "Black box denotes a matched "
            "algorithm-detector pair"
        ),
        "png_file": str(
            png_path.relative_to(repository_root)
        ).replace("\\", "/"),
        "pdf_file": str(
            pdf_path.relative_to(repository_root)
        ).replace("\\", "/"),
        "diagonal_intervals": {
            algorithm: {
                "q10": EXPECTED_DIAGONAL[algorithm]["q10"],
                "q90": EXPECTED_DIAGONAL[algorithm]["q90"],
            }
            for algorithm in ALGORITHM_ORDER
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=True,
        sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=" * 78)
    print("HSNNET Q10-Q90 HEATMAP GENERATION")
    print("=" * 78)
    print("Source       :", source_path)
    print("Source SHA256:", metadata["source_sha256"])
    print("Validated rows:", len(work))
    print("Validated pairs:", metadata["validated_pairs"])
    print("PNG          :", png_path)
    print("PDF          :", pdf_path)
    print("Metadata     :", metadata_path)
    print()
    print(
        "FINAL STATUS: PASS — THE REAL 5x5 "
        "Q10-Q90 SIGNATURE HEATMAP WAS GENERATED."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
