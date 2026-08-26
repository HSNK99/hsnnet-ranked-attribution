from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes() -> tuple[int, list[str]]:
    checksum_file = (
        ROOT
        / "manifests"
        / "SHA256SUMS.txt"
    )

    if not checksum_file.is_file():
        raise FileNotFoundError(
            checksum_file
        )

    failures = []
    checked = 0

    for line in checksum_file.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        if not line.strip():
            continue

        if "  " not in line:
            failures.append(
                f"MALFORMED | {line[:160]}"
            )
            continue

        digest, relative = line.split(
            "  ",
            1,
        )

        path = ROOT / relative

        if not path.is_file():
            failures.append(
                f"MISSING | {relative}"
            )
            continue

        observed = sha256_file(path)
        checked += 1

        if (
            observed.lower()
            != digest.lower()
        ):
            failures.append(
                f"HASH | {relative}"
            )

    return checked, failures


def privacy_scan() -> list[str]:
    findings = []

    # Match actual absolute Windows user paths.
    # This does not match escaped regex literals such as C:\\Users\\.
    absolute_user_path = re.compile(
        r"(?<!\\)[A-Za-z]:\\Users\\[^\\\r\n]+",
        re.IGNORECASE,
    )

    credential_patterns = [
        re.compile(
            r"\bghp_[A-Za-z0-9]{20,}\b"
        ),
        re.compile(
            r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
        ),
        re.compile(
            r"\bsk-[A-Za-z0-9_-]{20,}\b"
        ),
    ]

    text_suffixes = {
        ".py",
        ".json",
        ".csv",
        ".txt",
        ".md",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".cff",
    }

    excluded_names = {
        "SHA256SUMS.txt",
    }

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(ROOT)

        if (
            relative.parts
            and relative.parts[0]
            == "generated"
        ):
            continue

        if path.name in excluded_names:
            continue

        if (
            path.suffix.lower()
            not in text_suffixes
        ):
            continue

        if (
            path.stat().st_size
            > 30 * 1024 * 1024
        ):
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for match in (
            absolute_user_path.finditer(text)
        ):
            findings.append(
                f"{relative} | "
                f"absolute path: "
                f"{match.group(0)[:160]}"
            )

        for pattern in credential_patterns:
            if pattern.search(text):
                findings.append(
                    f"{relative} | "
                    "credential-like token"
                )

    return findings


def dataset_image_scan() -> list[str]:
    forbidden_suffixes = {
        ".pgm",
        ".ppm",
        ".bmp",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
    }

    return [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in forbidden_suffixes
        )
    ]


def required_item_scan() -> list[str]:
    required = [
        "README.md",
        "DATA_AVAILABILITY.md",
        "CITATION.cff",
        "LICENSE",
        "configs/experiment_locked.json",
        "configs/baselines_locked.json",
        "configs/paths.example.json",
        "environment/requirements-minimum.txt",
        "environment/environment.yml",
        "src/model_definition_hsnnet_locked.py",
        "scripts/repro_utils.py",
        "scripts/regenerate_tables.py",
        "scripts/verify_release.py",
        "scripts/verify_splits.py",
        "scripts/verify_checkpoints.py",
        "results/core/bossbase_04",
        "results/core/bossbase_02",
        "results/core/bows2_04",
        "results/core/bows2_02",
        "results/risk_conformal",
        "manifests/checkpoint_inventory.csv",
        "manifests/reviewer_requirements_matrix.csv",
        "manifests/SHA256SUMS.txt",
    ]

    return [
        item
        for item in required
        if not (ROOT / item).exists()
    ]


def run_regeneration() -> str:
    process = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "regenerate_tables.py"
            ),
            "--output",
            str(GENERATED),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (
        (process.stdout or "")
        + (process.stderr or "")
    )

    print(output)

    if process.returncode != 0:
        raise RuntimeError(
            "regenerate_tables.py failed."
        )

    return output


def main() -> None:
    missing = required_item_scan()

    if missing:
        raise RuntimeError(
            "Required repository items "
            "are missing:\n"
            + "\n".join(missing)
        )

    checked, hash_failures = (
        verify_hashes()
    )

    privacy_findings = privacy_scan()
    dataset_images = (
        dataset_image_scan()
    )

    diagnostics = {
        "checksummed_files": checked,
        "hash_failures": hash_failures,
        "privacy_findings": (
            privacy_findings
        ),
        "dataset_images": dataset_images,
    }

    GENERATED.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        GENERATED
        / "verify_release_diagnostics.json"
    ).write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if hash_failures:
        raise RuntimeError(
            "Checksum verification failed:\n"
            + "\n".join(
                hash_failures[:30]
            )
        )

    if privacy_findings:
        raise RuntimeError(
            "Personal path or credential "
            "was found:\n"
            + "\n".join(
                privacy_findings[:30]
            )
        )

    if dataset_images:
        raise RuntimeError(
            "Dataset images must not be "
            "redistributed:\n"
            + "\n".join(
                dataset_images[:30]
            )
        )

    run_regeneration()

    summary_path = (
        GENERATED
        / "regeneration_summary.json"
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if summary.get("status") != "PASS":
        raise RuntimeError(
            "Regeneration summary did "
            "not pass."
        )

    report = "\n".join([
        "HSNNET RELEASE VERIFICATION V2",
        "=" * 78,
        (
            "Checksummed files verified  : "
            f"{checked}"
        ),
        "Personal-path scan         : PASS",
        "Credential-pattern scan    : PASS",
        "Dataset-image scan         : PASS",
        "Four locked conditions     : PASS",
        "Table 9a                   : PASS",
        "201-point gamma sweep      : PASS",
        "Table 9b / Mondrian        : PASS",
        "Fig. 3 regeneration inputs : PASS",
        "",
        (
            "FINAL STATUS: PASS — "
            "ARCHIVED RESULTS REPRODUCED."
        ),
    ])

    (
        GENERATED
        / "VERIFY_RELEASE_REPORT.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
