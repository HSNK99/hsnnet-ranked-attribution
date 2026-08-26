from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [
    ROOT / "splits",
    ROOT / "configs",
    ROOT / "manifests",
    ROOT / "results",
]
OUT_DIR = ROOT / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTS = {
    "train": 7000,
    "validation": 1500,
    "test": 1500,
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "te": "test",
}

ID_HINTS = {
    "imageid", "pairid", "basename", "filename", "image",
    "id", "name", "file", "path", "coverid", "sampleid",
}

SPLIT_HINTS = {
    "split", "partition", "subset", "set", "foldname",
    "phase", "usage", "group",
}


def canonical(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalized_id(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    text = str(value).strip().strip("'").strip('"')
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    text = text.replace("\\", "/")
    name = Path(text).name

    # Collapse algorithm/payload prefixes when present and preserve pair identity.
    stem = Path(name).stem.lower()
    suffix = Path(name).suffix.lower()

    # Common image identifiers: 0001.pgm, 1.pgm, image_0001.pgm.
    match = re.search(r"(\d{1,8})$", stem)
    if match:
        number = match.group(1).lstrip("0") or "0"
        return f"{number}{suffix or '.pgm'}"

    return name.lower()


def split_from_text(value: Any) -> str | None:
    text = canonical(value)

    if text in SPLIT_ALIASES:
        return SPLIT_ALIASES[text]

    for token, normalized in SPLIT_ALIASES.items():
        if text.startswith(token) or text.endswith(token):
            return normalized

    return None


def extract_scalar_id(item: Any) -> str | None:
    if isinstance(item, (str, int, float)):
        return normalized_id(item)

    if isinstance(item, dict):
        normalized_keys = {
            canonical(key): key
            for key in item
        }

        for hint in ID_HINTS:
            if hint in normalized_keys:
                return normalized_id(
                    item[normalized_keys[hint]]
                )

        # Some manifests store one scalar value under an arbitrary key.
        scalar_values = [
            value
            for value in item.values()
            if isinstance(value, (str, int, float))
        ]

        if len(scalar_values) == 1:
            return normalized_id(scalar_values[0])

    return None


def values_to_ids(value: Any) -> set[str]:
    output: set[str] = set()

    if isinstance(value, dict):
        # Dict keyed by image ID.
        if value and all(
            isinstance(key, (str, int))
            for key in value.keys()
        ):
            keys_as_ids = {
                normalized_id(key)
                for key in value.keys()
            }
            keys_as_ids.discard(None)

            if len(keys_as_ids) >= 100:
                output.update(keys_as_ids)

        for item in value.values():
            output.update(values_to_ids(item))

    elif isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            scalar_id = extract_scalar_id(item)

            if scalar_id is not None:
                output.add(scalar_id)
            else:
                output.update(values_to_ids(item))

    else:
        scalar_id = extract_scalar_id(value)

        if scalar_id is not None:
            output.add(scalar_id)

    return output


def evaluate(label: str, groups: dict[str, set[str]]) -> dict[str, Any]:
    train = set(groups.get("train", set()))
    validation = set(groups.get("validation", set()))
    test = set(groups.get("test", set()))

    intersections = {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }

    counts = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }

    total_unique = len(
        train | validation | test
    )

    exact_counts = counts == TARGET_COUNTS
    no_overlap = all(
        count == 0
        for count in intersections.values()
    )

    complete = (
        exact_counts
        and no_overlap
        and total_unique == 10000
    )

    count_distance = sum(
        abs(counts[key] - TARGET_COUNTS[key])
        for key in TARGET_COUNTS
    )

    return {
        "label": label,
        **counts,
        "total_unique": total_unique,
        **intersections,
        "exact_counts": exact_counts,
        "no_overlap": no_overlap,
        "count_distance": count_distance,
        "status": "PASS" if complete else "FAIL",
    }


def detect_id_column(
    frame: pd.DataFrame,
) -> str | None:
    normalized = {
        canonical(column): column
        for column in frame.columns
    }

    for hint in ID_HINTS:
        if hint in normalized:
            return normalized[hint]

    # Prefer text-like columns containing image filenames.
    candidates = []

    for column in frame.columns:
        series = frame[column].dropna()

        if series.empty:
            continue

        sample = series.astype(str).head(1000)

        filename_share = sample.str.contains(
            r"\.(?:pgm|png|bmp|jpg|jpeg|tif|tiff)$",
            case=False,
            regex=True,
        ).mean()

        unique_share = sample.nunique() / max(1, len(sample))

        score = filename_share * 10 + unique_share

        if score > 0.8:
            candidates.append((score, column))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Last resort: first mostly unique column.
    for column in frame.columns:
        series = frame[column].dropna()

        if len(series) >= 1000:
            unique_share = (
                series.astype(str).nunique()
                / len(series)
            )

            if unique_share > 0.95:
                return column

    return None


def detect_split_column(
    frame: pd.DataFrame,
) -> str | None:
    normalized = {
        canonical(column): column
        for column in frame.columns
    }

    for hint in SPLIT_HINTS:
        if hint in normalized:
            return normalized[hint]

    for column in frame.columns:
        values = {
            split_from_text(value)
            for value in frame[column]
            .dropna()
            .astype(str)
            .head(10000)
        }

        values.discard(None)

        if len(values) >= 2:
            return column

    return None


def inspect_dataframe(
    frame: pd.DataFrame,
    label: str,
) -> list[dict[str, Any]]:
    results = []

    if frame.empty:
        return results

    # Case 1: row-wise image_id + split.
    split_column = detect_split_column(frame)
    id_column = detect_id_column(frame)

    if split_column is not None and id_column is not None:
        groups = defaultdict(set)

        for split_value, identity in zip(
            frame[split_column],
            frame[id_column],
        ):
            split = split_from_text(split_value)
            identity_value = normalized_id(identity)

            if split and identity_value:
                groups[split].add(identity_value)

        if groups:
            results.append({
                "label": (
                    f"{label} "
                    f"[row-wise: {id_column} + {split_column}]"
                ),
                "groups": groups,
            })

    # Case 2: wide columns such as train_ids, validation_files, test_pairs.
    wide_groups = defaultdict(set)

    for column in frame.columns:
        split = split_from_text(column)

        if split is None:
            continue

        for value in frame[column].dropna():
            identity_value = normalized_id(value)

            if identity_value:
                wide_groups[split].add(identity_value)

    if wide_groups:
        results.append({
            "label": f"{label} [wide split columns]",
            "groups": wide_groups,
        })

    # Case 3: one split per file, inferred from filename.
    file_split = split_from_text(
        Path(label).stem
    )

    if file_split is not None:
        inferred_id_column = (
            id_column
            if id_column is not None
            else frame.columns[0]
        )

        identities = {
            normalized_id(value)
            for value in frame[
                inferred_id_column
            ].dropna()
        }

        identities.discard(None)

        if identities:
            results.append({
                "label": label,
                "single_split": file_split,
                "ids": identities,
                "parent_key": str(
                    Path(label).parent
                ),
            })

    return results


def read_table_variants(
    path: Path,
) -> list[pd.DataFrame]:
    frames = []

    if path.suffix.lower() == ".csv":
        for header in [0, None]:
            try:
                frame = pd.read_csv(
                    path,
                    low_memory=False,
                    header=header,
                )

                if header is None:
                    frame.columns = [
                        f"column_{index}"
                        for index in range(
                            len(frame.columns)
                        )
                    ]

                frames.append(frame)
            except Exception:
                pass

    elif path.suffix.lower() in {
        ".xls",
        ".xlsx",
    }:
        try:
            sheets = pd.read_excel(
                path,
                sheet_name=None,
            )

            frames.extend(
                sheets.values()
            )
        except Exception:
            pass

    return frames


def json_candidates(
    value: Any,
    label: str,
) -> list[dict[str, Any]]:
    output = []

    if isinstance(value, dict):
        groups = defaultdict(set)

        for key, item in value.items():
            split = split_from_text(key)

            if split:
                groups[split].update(
                    values_to_ids(item)
                )

        if groups:
            output.append({
                "label": f"{label} [JSON split keys]",
                "groups": groups,
            })

        # List of records with split and ID fields.
        if value:
            normalized_keys = {
                canonical(key): key
                for key in value
            }

            split_key = next(
                (
                    normalized_keys[hint]
                    for hint in SPLIT_HINTS
                    if hint in normalized_keys
                ),
                None,
            )

            id_key = next(
                (
                    normalized_keys[hint]
                    for hint in ID_HINTS
                    if hint in normalized_keys
                ),
                None,
            )

            if split_key and id_key:
                split = split_from_text(
                    value[split_key]
                )
                identity_value = normalized_id(
                    value[id_key]
                )

                if split and identity_value:
                    output.append({
                        "label": (
                            f"{label} "
                            "[JSON record]"
                        ),
                        "single_record": (
                            split,
                            identity_value,
                        ),
                    })

        for key, item in value.items():
            output.extend(
                json_candidates(
                    item,
                    f"{label}.{key}",
                )
            )

    elif isinstance(value, list):
        record_groups = defaultdict(set)

        for index, item in enumerate(value):
            if isinstance(item, dict):
                normalized_keys = {
                    canonical(key): key
                    for key in item
                }

                split_key = next(
                    (
                        normalized_keys[hint]
                        for hint in SPLIT_HINTS
                        if hint in normalized_keys
                    ),
                    None,
                )

                id_key = next(
                    (
                        normalized_keys[hint]
                        for hint in ID_HINTS
                        if hint in normalized_keys
                    ),
                    None,
                )

                if split_key and id_key:
                    split = split_from_text(
                        item[split_key]
                    )
                    identity_value = normalized_id(
                        item[id_key]
                    )

                    if split and identity_value:
                        record_groups[split].add(
                            identity_value
                        )

            output.extend(
                json_candidates(
                    item,
                    f"{label}[{index}]",
                )
            )

        if record_groups:
            output.append({
                "label": f"{label} [JSON records]",
                "groups": record_groups,
            })

    return output


def inspect_json(
    path: Path,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except Exception:
        return []

    return json_candidates(
        value,
        str(path.relative_to(ROOT)),
    )


def inspect_text(
    path: Path,
) -> list[dict[str, Any]]:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return []

    # Dictionary or list embedded in a Python/text file.
    output = []

    for variable_name, split in [
        ("train", "train"),
        ("training", "train"),
        ("val", "validation"),
        ("valid", "validation"),
        ("validation", "validation"),
        ("test", "test"),
    ]:
        pattern = re.compile(
            rf"\b{variable_name}\w*\s*=\s*(\[[\s\S]*?\])",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            try:
                value = ast.literal_eval(
                    match.group(1)
                )
            except Exception:
                continue

            identities = values_to_ids(value)

            if identities:
                output.append({
                    "label": (
                        f"{path.relative_to(ROOT)} "
                        f"[text variable {variable_name}]"
                    ),
                    "single_split": split,
                    "ids": identities,
                    "parent_key": str(
                        path.parent.relative_to(ROOT)
                    ),
                })

    return output


def all_candidate_files() -> Iterable[Path]:
    allowed_suffixes = {
        ".csv",
        ".xls",
        ".xlsx",
        ".json",
        ".txt",
        ".py",
    }

    seen = set()

    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower()
                in allowed_suffixes
            ):
                resolved = path.resolve()

                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def main() -> None:
    combined_candidates = []
    separate_groups = defaultdict(
        lambda: defaultdict(set)
    )
    json_record_groups = defaultdict(
        lambda: defaultdict(set)
    )

    examined_files = 0

    for path in all_candidate_files():
        examined_files += 1
        suffix = path.suffix.lower()

        if suffix in {
            ".csv",
            ".xls",
            ".xlsx",
        }:
            for sheet_index, frame in enumerate(
                read_table_variants(path)
            ):
                label = str(
                    path.relative_to(ROOT)
                )

                if suffix in {
                    ".xls",
                    ".xlsx",
                }:
                    label += (
                        f" [sheet {sheet_index}]"
                    )

                for candidate in inspect_dataframe(
                    frame,
                    label,
                ):
                    if "groups" in candidate:
                        combined_candidates.append(
                            candidate
                        )
                    else:
                        separate_groups[
                            candidate["parent_key"]
                        ][
                            candidate["single_split"]
                        ].update(
                            candidate["ids"]
                        )

        elif suffix == ".json":
            for candidate in inspect_json(path):
                if "groups" in candidate:
                    combined_candidates.append(
                        candidate
                    )
                elif "single_record" in candidate:
                    split, identity_value = (
                        candidate["single_record"]
                    )

                    json_record_groups[
                        str(path.relative_to(ROOT))
                    ][split].add(
                        identity_value
                    )

        elif suffix in {
            ".txt",
            ".py",
        }:
            for candidate in inspect_text(path):
                separate_groups[
                    candidate["parent_key"]
                ][
                    candidate["single_split"]
                ].update(
                    candidate["ids"]
                )

    for parent_key, groups in (
        separate_groups.items()
    ):
        combined_candidates.append({
            "label": (
                f"{parent_key} "
                "[combined separate split files]"
            ),
            "groups": groups,
        })

    for label, groups in (
        json_record_groups.items()
    ):
        combined_candidates.append({
            "label": (
                f"{label} "
                "[combined JSON records]"
            ),
            "groups": groups,
        })

    audit_rows = [
        evaluate(
            candidate["label"],
            candidate["groups"],
        )
        for candidate in combined_candidates
    ]

    audit = pd.DataFrame(audit_rows)

    if audit.empty:
        raise RuntimeError(
            "No machine-readable split representation "
            "was detected."
        )

    audit = audit.sort_values(
        [
            "status",
            "count_distance",
            "total_unique",
            "label",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    audit.to_csv(
        OUT_DIR / "split_leakage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    passing = audit.loc[
        audit["status"].eq("PASS")
    ]

    report_lines = [
        "HSNNET SPLIT LEAKAGE AUDIT V2",
        "=" * 78,
        f"Files examined                  : {examined_files}",
        f"Candidate representations       : {len(audit)}",
        f"Complete passing representations: {len(passing)}",
        "",
    ]

    if not passing.empty:
        best = passing.iloc[0]

        report_lines += [
            f"Selected evidence : {best['label']}",
            f"Train identities  : {int(best['train']):,}",
            f"Validation IDs    : {int(best['validation']):,}",
            f"Test identities   : {int(best['test']):,}",
            "train ∩ validation: 0",
            "train ∩ test      : 0",
            "validation ∩ test : 0",
            "",
            "FINAL STATUS: PASS — PAIR-IDENTITY SPLIT IS LEAKAGE-SAFE.",
        ]

    else:
        report_lines += [
            "Closest detected representations:",
            "-" * 78,
            audit.head(20).to_string(
                index=False
            ),
            "",
            "FINAL STATUS: FAIL — EXACT 7000/1500/1500 "
            "PAIR-ID EVIDENCE WAS NOT FOUND.",
        ]

    report = "\n".join(report_lines)

    (
        OUT_DIR
        / "VERIFY_SPLITS_REPORT.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    if passing.empty:
        raise RuntimeError(
            "Exact pair-ID split evidence is incomplete. "
            "Review generated/split_leakage_audit.csv."
        )


if __name__ == "__main__":
    main()
