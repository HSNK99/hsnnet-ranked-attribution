#!/usr/bin/env python
"""
Generate the locked risk-coverage and gamma-sweep curves for the
HSNNET ranked-attribution reproducibility package.

The figures are generated only from archived verified CSV files.
No detector inference, model fitting, recalibration, or target-domain
parameter selection is performed.
"""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCKED_GAMMA = 0.900

EXPECTED_PROPOSED = {
    "retention_percent": 91.262,
    "mean_set_size": 3.75866,
    "search_reduction_percent": 24.8268,
    "empty_percent": 1.694,
    "decision_coverage_percent": 98.306,
    "conditional_false_exclusion_percent": 7.165382,
}

GAMMA_REQUIRED_COLUMNS = {
    "gamma",
    "retention_percent",
    "mean_set_size",
    "search_reduction_percent",
    "empty_percent",
    "decision_coverage_percent",
    "conditional_retention_percent",
    "conditional_false_exclusion_percent",
}

WORKLOAD_REQUIRED_COLUMNS = {
    "max_set_k",
    "accepted_n",
    "coverage_percent",
    "conditional_retention_percent",
    "conditional_false_exclusion_percent",
    "mean_admitted_size",
    "expected_accepted_n",
    "count_status",
}

COMPARISON_REQUIRED_COLUMNS = {
    "method",
    "retention_percent",
    "mean_set_size",
    "search_reduction_percent",
    "empty_percent",
    "decision_coverage_percent",
    "conditional_false_exclusion_percent",
    "calibration",
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


def require_columns(
    frame: pd.DataFrame,
    required: set,
    label: str,
) -> None:
    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{label} is missing columns: "
            + ", ".join(sorted(missing))
        )


def convert_numeric(
    frame: pd.DataFrame,
    columns,
    label: str,
) -> pd.DataFrame:
    work = frame.copy()

    for column in columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

        if work[column].isna().any():
            raise RuntimeError(
                f"{label} contains nonnumeric values "
                f"in column {column}."
            )

    return work


def validate_gamma(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        GAMMA_REQUIRED_COLUMNS,
        "Gamma-sweep table",
    )

    numeric_columns = sorted(
        GAMMA_REQUIRED_COLUMNS
    )

    work = convert_numeric(
        frame,
        numeric_columns,
        "Gamma-sweep table",
    )

    if len(work) != 201:
        raise RuntimeError(
            f"Expected 201 gamma rows, found {len(work)}."
        )

    work = work.sort_values(
        "gamma"
    ).reset_index(drop=True)

    gamma = work["gamma"].to_numpy(
        dtype=float
    )

    if not np.isclose(
        gamma[0],
        0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Gamma sweep does not begin at 0.000."
        )

    if not np.isclose(
        gamma[-1],
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Gamma sweep does not end at 1.000."
        )

    if len(np.unique(gamma)) != 201:
        raise RuntimeError(
            "Gamma values are not unique."
        )

    if not np.allclose(
        np.diff(gamma),
        0.005,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Gamma increments are not uniformly 0.005."
        )

    locked_rows = work[
        np.isclose(
            work["gamma"],
            LOCKED_GAMMA,
            atol=1e-12,
            rtol=0.0,
        )
    ]

    if len(locked_rows) != 1:
        raise RuntimeError(
            "The locked gamma=0.900 row is missing "
            "or duplicated."
        )

    return work


def validate_workload(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        WORKLOAD_REQUIRED_COLUMNS,
        "Workload-conditioned table",
    )

    numeric_columns = [
        column
        for column in WORKLOAD_REQUIRED_COLUMNS
        if column != "count_status"
    ]

    work = convert_numeric(
        frame,
        numeric_columns,
        "Workload-conditioned table",
    )

    if len(work) != 5:
        raise RuntimeError(
            f"Expected five workload rows, found {len(work)}."
        )

    work = work.sort_values(
        "max_set_k"
    ).reset_index(drop=True)

    if work["max_set_k"].astype(int).tolist() != [
        1, 2, 3, 4, 5
    ]:
        raise RuntimeError(
            "Expected workload thresholds k=1...5."
        )

    if not (
        frame["count_status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
        .all()
    ):
        raise RuntimeError(
            "At least one workload count check did not pass."
        )

    if not np.allclose(
        work["accepted_n"],
        work["expected_accepted_n"],
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Accepted counts differ from expected counts."
        )

    return work


def validate_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        COMPARISON_REQUIRED_COLUMNS,
        "Proposed-Mondrian comparison",
    )

    numeric_columns = [
        column
        for column in COMPARISON_REQUIRED_COLUMNS
        if column not in {
            "method",
            "calibration",
        }
    ]

    work = convert_numeric(
        frame,
        numeric_columns,
        "Proposed-Mondrian comparison",
    )

    if len(work) != 2:
        raise RuntimeError(
            f"Expected two comparison rows, found {len(work)}."
        )

    method_text = (
        work["method"]
        .astype(str)
        .str.lower()
    )

    proposed_rows = work[
        method_text.str.contains(
            "proposed",
            regex=False,
        )
    ]

    mondrian_rows = work[
        method_text.str.contains(
            "mondrian",
            regex=False,
        )
    ]

    if len(proposed_rows) != 1:
        raise RuntimeError(
            "Expected exactly one proposed-method row."
        )

    if len(mondrian_rows) != 1:
        raise RuntimeError(
            "Expected exactly one Mondrian row."
        )

    proposed = proposed_rows.iloc[0]

    for column, expected_value in EXPECTED_PROPOSED.items():
        actual_value = float(proposed[column])

        if not np.isclose(
            actual_value,
            expected_value,
            atol=5e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Unexpected proposed value for {column}: "
                f"{actual_value}"
            )

    return work


def validate_locked_consistency(
    gamma: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    gamma_row = gamma[
        np.isclose(
            gamma["gamma"],
            LOCKED_GAMMA,
            atol=1e-12,
            rtol=0.0,
        )
    ].iloc[0]

    proposed_row = comparison[
        comparison["method"]
        .astype(str)
        .str.lower()
        .str.contains(
            "proposed",
            regex=False,
        )
    ].iloc[0]

    shared_columns = [
        "retention_percent",
        "mean_set_size",
        "search_reduction_percent",
        "empty_percent",
        "decision_coverage_percent",
        "conditional_false_exclusion_percent",
    ]

    for column in shared_columns:
        if not np.isclose(
            float(gamma_row[column]),
            float(proposed_row[column]),
            atol=5e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Locked gamma sweep and comparison table "
                f"disagree for {column}."
            )


def save_figure(
    figure,
    png_path: Path,
    pdf_path: Path,
    title: str,
    subject: str,
) -> None:
    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Description": subject,
        },
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "HSNNET reproducibility package",
            "Subject": subject,
            "CreationDate": None,
            "ModDate": None,
        },
    )

    plt.close(figure)


def generate_gamma_risk_coverage(
    gamma: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    locked = gamma[
        np.isclose(
            gamma["gamma"],
            LOCKED_GAMMA,
            atol=1e-12,
            rtol=0.0,
        )
    ].iloc[0]

    figure, axis = plt.subplots(
        figsize=(9.6, 6.4),
        constrained_layout=True,
    )

    axis.plot(
        gamma["gamma"],
        gamma["decision_coverage_percent"],
        linewidth=2.0,
        label="Decision coverage",
    )

    axis.plot(
        gamma["gamma"],
        gamma[
            "conditional_false_exclusion_percent"
        ],
        linewidth=2.0,
        label="Conditional false-exclusion risk",
    )

    axis.axvline(
        LOCKED_GAMMA,
        linestyle="--",
        linewidth=1.7,
        label="Locked gamma = 0.900",
    )

    axis.scatter(
        [LOCKED_GAMMA],
        [locked["decision_coverage_percent"]],
        marker="o",
        s=65,
        zorder=4,
    )

    axis.scatter(
        [LOCKED_GAMMA],
        [
            locked[
                "conditional_false_exclusion_percent"
            ]
        ],
        marker="s",
        s=65,
        zorder=4,
    )

    axis.annotate(
        (
            f"Coverage = "
            f"{locked['decision_coverage_percent']:.3f}%"
        ),
        (
            LOCKED_GAMMA,
            locked["decision_coverage_percent"],
        ),
        xytext=(-118, -28),
        textcoords="offset points",
        fontsize=10,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    axis.annotate(
        (
            f"Risk = "
            f"{locked['conditional_false_exclusion_percent']:.3f}%"
        ),
        (
            LOCKED_GAMMA,
            locked[
                "conditional_false_exclusion_percent"
            ],
        ),
        xytext=(-102, 28),
        textcoords="offset points",
        fontsize=10,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    axis.set_xlabel(
        "Gamma threshold",
        fontsize=12,
    )

    axis.set_ylabel(
        "Percentage (%)",
        fontsize=12,
    )

    axis.set_title(
        "Gamma-sweep decision coverage and conditional risk",
        fontsize=14,
        pad=12,
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        bottom=0.0,
    )

    axis.grid(
        True,
        alpha=0.30,
    )

    axis.legend(
        fontsize=10,
        loc="best",
    )

    save_figure(
        figure,
        png_path,
        pdf_path,
        (
            "HSNNET gamma-sweep decision coverage "
            "and conditional risk"
        ),
        (
            "Decision coverage and conditional "
            "false-exclusion risk across 201 gamma values."
        ),
    )


def generate_gamma_workload_tradeoff(
    gamma: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    locked = gamma[
        np.isclose(
            gamma["gamma"],
            LOCKED_GAMMA,
            atol=1e-12,
            rtol=0.0,
        )
    ].iloc[0]

    figure, left_axis = plt.subplots(
        figsize=(9.6, 6.4),
        constrained_layout=True,
    )

    right_axis = left_axis.twinx()

    line_set_size = left_axis.plot(
        gamma["gamma"],
        gamma["mean_set_size"],
        linewidth=2.0,
        label="Mean candidate-set size",
    )[0]

    line_reduction = right_axis.plot(
        gamma["gamma"],
        gamma["search_reduction_percent"],
        linewidth=2.0,
        linestyle="--",
        label="Search-space reduction",
    )[0]

    locked_line = left_axis.axvline(
        LOCKED_GAMMA,
        linestyle=":",
        linewidth=1.7,
        label="Locked gamma = 0.900",
    )

    left_axis.scatter(
        [LOCKED_GAMMA],
        [locked["mean_set_size"]],
        marker="o",
        s=65,
        zorder=4,
    )

    right_axis.scatter(
        [LOCKED_GAMMA],
        [locked["search_reduction_percent"]],
        marker="s",
        s=65,
        zorder=4,
    )

    left_axis.annotate(
        (
            f"Mean size = "
            f"{locked['mean_set_size']:.3f}"
        ),
        (
            LOCKED_GAMMA,
            locked["mean_set_size"],
        ),
        xytext=(-112, -38),
        textcoords="offset points",
        fontsize=10,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    right_axis.annotate(
        (
            f"Reduction = "
            f"{locked['search_reduction_percent']:.3f}%"
        ),
        (
            LOCKED_GAMMA,
            locked["search_reduction_percent"],
        ),
        xytext=(-128, 36),
        textcoords="offset points",
        fontsize=10,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    left_axis.set_xlabel(
        "Gamma threshold",
        fontsize=12,
    )

    left_axis.set_ylabel(
        "Mean candidate-set size",
        fontsize=12,
    )

    right_axis.set_ylabel(
        "Search-space reduction (%)",
        fontsize=12,
    )

    left_axis.set_title(
        "Gamma-sweep workload and search-space trade-off",
        fontsize=14,
        pad=12,
    )

    left_axis.set_xlim(
        0.0,
        1.0,
    )

    left_axis.set_ylim(
        0.0,
        5.25,
    )

    right_axis.set_ylim(
        bottom=0.0,
    )

    left_axis.grid(
        True,
        alpha=0.30,
    )

    left_axis.legend(
        handles=[
            line_set_size,
            line_reduction,
            locked_line,
        ],
        fontsize=10,
        loc="best",
    )

    save_figure(
        figure,
        png_path,
        pdf_path,
        (
            "HSNNET gamma-sweep workload and "
            "search-space trade-off"
        ),
        (
            "Mean candidate-set size and search-space "
            "reduction across 201 gamma values."
        ),
    )


def generate_workload_risk_coverage(
    workload: pd.DataFrame,
    comparison: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9.6, 6.4),
        constrained_layout=True,
    )

    axis.plot(
        workload["coverage_percent"],
        workload[
            "conditional_false_exclusion_percent"
        ],
        marker="o",
        linewidth=2.0,
        markersize=7,
        label="Workload-conditioned curve",
    )

    for _, row in workload.iterrows():
        k_value = int(row["max_set_k"])

        # k=5 coincides exactly with the proposed locked
        # operating point and is labelled jointly below.
        if k_value == 5:
            continue

        axis.annotate(
            f"k={k_value}",
            (
                row["coverage_percent"],
                row[
                    "conditional_false_exclusion_percent"
                ],
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )

    method_text = (
        comparison["method"]
        .astype(str)
        .str.lower()
    )

    proposed = comparison[
        method_text.str.contains(
            "proposed",
            regex=False,
        )
    ].iloc[0]

    mondrian = comparison[
        method_text.str.contains(
            "mondrian",
            regex=False,
        )
    ].iloc[0]

    axis.scatter(
        [proposed["decision_coverage_percent"]],
        [
            proposed[
                "conditional_false_exclusion_percent"
            ]
        ],
        marker="s",
        s=85,
        label="Proposed, frozen gamma=0.900",
        zorder=5,
    )

    axis.scatter(
        [mondrian["decision_coverage_percent"]],
        [
            mondrian[
                "conditional_false_exclusion_percent"
            ]
        ],
        marker="D",
        s=85,
        label="Two-fold Mondrian, alpha=0.10",
        zorder=5,
    )

    axis.annotate(
        (
            f"Proposed (k=5)\n"
            f"({proposed['decision_coverage_percent']:.3f}%, "
            f"{proposed['conditional_false_exclusion_percent']:.3f}%)"
        ),
        (
            proposed["decision_coverage_percent"],
            proposed[
                "conditional_false_exclusion_percent"
            ],
        ),
        xytext=(-155, 45),
        textcoords="offset points",
        fontsize=9.5,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    axis.annotate(
        (
            f"Mondrian\n"
            f"({mondrian['decision_coverage_percent']:.3f}%, "
            f"{mondrian['conditional_false_exclusion_percent']:.3f}%)"
        ),
        (
            mondrian["decision_coverage_percent"],
            mondrian[
                "conditional_false_exclusion_percent"
            ],
        ),
        xytext=(-175, -25),
        textcoords="offset points",
        fontsize=9.5,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
    )

    axis.set_xlabel(
        "Decision coverage (%)",
        fontsize=12,
    )

    axis.set_ylabel(
        "Conditional false-exclusion risk (%)",
        fontsize=12,
    )

    axis.set_title(
        "Workload-conditioned risk-coverage diagnostic",
        fontsize=14,
        pad=12,
    )

    axis.set_xlim(
        0.0,
        102.0,
    )

    axis.set_ylim(
        bottom=0.0,
    )

    axis.grid(
        True,
        alpha=0.30,
    )

    axis.legend(
        fontsize=9.5,
        loc="best",
    )

    save_figure(
        figure,
        png_path,
        pdf_path,
        (
            "HSNNET workload-conditioned "
            "risk-coverage diagnostic"
        ),
        (
            "Risk-coverage-style diagnostic for workload "
            "thresholds k=1...5 with proposed and "
            "Mondrian operating points."
        ),
    )


def main() -> None:
    repository_root = (
        Path(__file__).resolve().parents[1]
    )

    source_directory = (
        repository_root
        / "results"
        / "risk_conformal"
        / "step6_verified"
    )

    gamma_path = (
        source_directory
        / "gamma_sweep_201_points.csv"
    )

    workload_path = (
        source_directory
        / "table_9a_workload_conditioned_risk_coverage.csv"
    )

    comparison_path = (
        source_directory
        / "table_9b_proposed_vs_mondrian.csv"
    )

    output_directory = (
        repository_root
        / "figures"
    )

    source_paths = [
        gamma_path,
        workload_path,
        comparison_path,
    ]

    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    gamma = validate_gamma(
        pd.read_csv(gamma_path)
    )

    workload = validate_workload(
        pd.read_csv(workload_path)
    )

    comparison = validate_comparison(
        pd.read_csv(comparison_path)
    )

    validate_locked_consistency(
        gamma,
        comparison,
    )

    gamma_risk_png = (
        output_directory
        / "gamma_sweep_decision_coverage_false_exclusion.png"
    )

    gamma_risk_pdf = (
        output_directory
        / "gamma_sweep_decision_coverage_false_exclusion.pdf"
    )

    gamma_workload_png = (
        output_directory
        / "gamma_sweep_set_size_search_reduction.png"
    )

    gamma_workload_pdf = (
        output_directory
        / "gamma_sweep_set_size_search_reduction.pdf"
    )

    workload_curve_png = (
        output_directory
        / "workload_conditioned_risk_coverage.png"
    )

    workload_curve_pdf = (
        output_directory
        / "workload_conditioned_risk_coverage.pdf"
    )

    metadata_path = (
        output_directory
        / "risk_coverage_curves_metadata.json"
    )

    generate_gamma_risk_coverage(
        gamma,
        gamma_risk_png,
        gamma_risk_pdf,
    )

    generate_gamma_workload_tradeoff(
        gamma,
        gamma_workload_png,
        gamma_workload_pdf,
    )

    generate_workload_risk_coverage(
        workload,
        comparison,
        workload_curve_png,
        workload_curve_pdf,
    )

    locked_gamma_row = gamma[
        np.isclose(
            gamma["gamma"],
            LOCKED_GAMMA,
            atol=1e-12,
            rtol=0.0,
        )
    ].iloc[0]

    metadata = {
        "dataset": "BOWS2",
        "payload_bpp": 0.4,
        "diagnostic_scope": (
            "Archived decision-layer risk-coverage "
            "and gamma-sweep diagnostics"
        ),
        "gamma_grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "increment": 0.005,
            "points": 201,
        },
        "locked_gamma": LOCKED_GAMMA,
        "locked_operating_point": {
            column: float(
                locked_gamma_row[column]
            )
            for column in [
                "retention_percent",
                "mean_set_size",
                "search_reduction_percent",
                "empty_percent",
                "decision_coverage_percent",
                "conditional_retention_percent",
                "conditional_false_exclusion_percent",
            ]
        },
        "workload_thresholds": (
            workload["max_set_k"]
            .astype(int)
            .tolist()
        ),
        "source_files": {
            str(
                path.relative_to(repository_root)
            ).replace("\\", "/"): {
                "sha256": sha256_file(path),
                "rows": int(
                    len(pd.read_csv(path))
                ),
            }
            for path in source_paths
        },
        "output_files": [
            str(
                path.relative_to(repository_root)
            ).replace("\\", "/")
            for path in [
                gamma_risk_png,
                gamma_risk_pdf,
                gamma_workload_png,
                gamma_workload_pdf,
                workload_curve_png,
                workload_curve_pdf,
            ]
        ],
        "interpretation_limit": (
            "These plots are descriptive diagnostics. "
            "They do not establish finite-sample "
            "conformal coverage guarantees."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=" * 84)
    print("HSNNET RISK-COVERAGE CURVE GENERATION")
    print("=" * 84)
    print("Gamma rows       :", len(gamma))
    print("Gamma minimum    :", gamma["gamma"].min())
    print("Gamma maximum    :", gamma["gamma"].max())
    print("Locked gamma     :", LOCKED_GAMMA)
    print("Workload points  :", len(workload))
    print("Comparison rows  :", len(comparison))
    print()
    print(
        "Locked coverage  :",
        f"{locked_gamma_row['decision_coverage_percent']:.6f}%",
    )
    print(
        "Locked risk      :",
        f"{locked_gamma_row['conditional_false_exclusion_percent']:.6f}%",
    )
    print(
        "Locked mean size :",
        f"{locked_gamma_row['mean_set_size']:.6f}",
    )
    print(
        "Locked reduction :",
        f"{locked_gamma_row['search_reduction_percent']:.6f}%",
    )
    print()
    print("Generated files:")

    for output_path in [
        gamma_risk_png,
        gamma_risk_pdf,
        gamma_workload_png,
        gamma_workload_pdf,
        workload_curve_png,
        workload_curve_pdf,
        metadata_path,
    ]:
        print(
            " ",
            output_path.relative_to(repository_root),
        )

    print()
    print(
        "FINAL STATUS: PASS — THREE LOCKED "
        "RISK-COVERAGE AND GAMMA-SWEEP FIGURES "
        "WERE GENERATED."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
