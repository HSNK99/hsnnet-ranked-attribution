from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


ALGOS = ["WOW", "S-UNIWARD", "HILL", "HUGO", "MiPOD"]
PAYLOADS = ["0.2bpp", "0.4bpp"]
EXPECTED_TOTAL_PARAMETERS = 7_293_009
EXPECTED_TRAINABLE_PARAMETERS = 7_292_609
EXPECTED_FRONTEND_PARAMETERS = 400
EXPECTED_IMAGE_SIZE = (256, 256)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def candidate_pipeline_paths() -> Iterable[Path]:
    cwd = Path.cwd()
    roots = [
        cwd,
        *cwd.parents,
        Path(r"${PROJECT_ROOT}"),
        Path(r"${USER_HOME}\Downloads"),
        Path(r"${USER_HOME}\Desktop"),
    ]
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except OSError:
            key = str(root).lower()
        if key in seen or not root.exists():
            continue
        seen.add(key)

        direct = [
            root / "scripts" / "run_candidate_pipeline_full5.py",
            root / "run_candidate_pipeline_full5.py",
            root / "42_EXBOW2_FULL5_EXTERNAL_READY" / "scripts" / "run_candidate_pipeline_full5.py",
            root / "42_EXBOW2_FULL5_EXTERNAL_READY_AUTOLOCATE" / "scripts" / "run_candidate_pipeline_full5.py",
        ]
        for path in direct:
            if path.is_file():
                yield path.resolve()


def find_pipeline(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline not found: {path}")
        return path.resolve()

    for path in candidate_pipeline_paths():
        return path

    raise FileNotFoundError(
        "Could not locate run_candidate_pipeline_full5.py. "
        "Pass it explicitly with --pipeline."
    )


def torch_load_state(torch, path: Path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def install_strict_dataset(pipeline):
    """
    Prevents silent resizing or RGB-to-grayscale conversion during bank
    reconstruction. Every image must already be 256 x 256 and 8-bit grayscale.
    """
    pipeline.import_torch()

    def factory():
        class StrictResponseDataset(pipeline.Dataset):
            def __init__(self, rows, image_size):
                self.rows = rows
                self.image_size = int(image_size)

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = self.rows[index]
                path = Path(row["path"])
                with Image.open(path) as image:
                    if image.mode != "L":
                        raise ValueError(
                            f"Non-grayscale image in strict bank run: "
                            f"{path} | mode={image.mode}"
                        )
                    if image.size != (self.image_size, self.image_size):
                        raise ValueError(
                            f"Image-size mismatch in strict bank run: "
                            f"{path} | size={image.size} | "
                            f"expected={(self.image_size, self.image_size)}. "
                            "No post-embedding resize is permitted."
                        )
                    arr = np.asarray(image, dtype=np.float32) / 255.0

                tensor = pipeline.torch.from_numpy(arr).unsqueeze(0)
                return tensor, int(index)

        return StrictResponseDataset

    pipeline.make_dataset_class = factory


def architecture_integrity_audit(
    pipeline,
    model_module,
    payloads: List[str],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    pipeline.import_torch()
    torch = pipeline.torch
    device = torch.device("cpu")
    rows = []

    model_hash = sha256_file(Path(model_module.__file__))

    for payload in payloads:
        manifest = pipeline.build_manifest(payload)

        for algo in ALGOS:
            meta = manifest[algo]
            checkpoint = Path(meta["checkpoint"])
            cfg = pipeline.build_cfg_obj(
                model_module,
                algo,
                payload,
                meta["run_config"],
            )

            # The bank reconstruction is strict and never resizes.
            cfg.use_resize = False

            model = model_module.hsnnet(cfg).to(device)
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(
                p.numel() for p in model.parameters()
                if p.requires_grad
            )
            frontend = sum(
                p.numel() for p in model.frontend.parameters()
            )

            raw = torch_load_state(torch, checkpoint, device)
            state = pipeline.clean_state_dict(raw)

            model_keys = set(model.state_dict().keys())
            state_keys = set(state.keys())
            missing = sorted(model_keys - state_keys)
            unexpected = sorted(state_keys - model_keys)

            shape_mismatches = []
            for key in sorted(model_keys & state_keys):
                expected_shape = tuple(model.state_dict()[key].shape)
                found_shape = tuple(state[key].shape)
                if expected_shape != found_shape:
                    shape_mismatches.append({
                        "key": key,
                        "expected": expected_shape,
                        "found": found_shape,
                    })

            strict_pass = (
                not missing
                and not unexpected
                and not shape_mismatches
            )

            error = ""
            output_shape = ""
            try:
                model.load_state_dict(state, strict=True)
                model.eval()
                with torch.no_grad():
                    output = model(
                        torch.zeros(
                            1,
                            1,
                            EXPECTED_IMAGE_SIZE[1],
                            EXPECTED_IMAGE_SIZE[0],
                            dtype=torch.float32,
                        )
                    )
                output_shape = str(tuple(output.shape))
                strict_pass = strict_pass and tuple(output.shape) == (1, 1)
            except Exception as exc:
                strict_pass = False
                error = repr(exc)

            count_pass = (
                total == EXPECTED_TOTAL_PARAMETERS
                and trainable == EXPECTED_TRAINABLE_PARAMETERS
                and frontend == EXPECTED_FRONTEND_PARAMETERS
            )

            row = {
                "payload": payload,
                "algorithm": algo,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "model_definition": str(Path(model_module.__file__)),
                "model_definition_sha256": model_hash,
                "total_parameters": int(total),
                "trainable_parameters": int(trainable),
                "frontend_parameters": int(frontend),
                "state_dict_keys_model": len(model_keys),
                "state_dict_keys_checkpoint": len(state_keys),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "shape_mismatches": len(shape_mismatches),
                "output_shape": output_shape,
                "strict_load_pass": bool(strict_pass),
                "parameter_count_pass": bool(count_pass),
                "status": "PASS" if strict_pass and count_pass else "FAIL",
                "error": error,
            }
            rows.append(row)

            print(
                f"[ARCH] {payload:6s} | {algo:11s} | "
                f"strict={row['strict_load_pass']} | "
                f"params={total:,} | {row['status']}",
                flush=True,
            )

            del model

    save_csv(rows, out_dir / "architecture_integrity_audit.csv")
    save_json(rows, out_dir / "architecture_integrity_audit.json")

    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(
            f"Architecture integrity audit failed for {len(failures)} "
            "detector checkpoint(s). See audit outputs."
        )
    return rows


def iter_stage_directories(pipeline, stage: str):
    payloads = []
    include_boss = False
    include_bows = False

    if stage in {"boss02", "bows02"}:
        payloads = ["0.2bpp"]
    elif stage in {"boss04", "bows04"}:
        payloads = ["0.4bpp"]
    elif stage in {"audit", "all", "table"}:
        payloads = PAYLOADS.copy()

    if stage in {"boss02", "boss04", "bows02", "bows04", "audit", "all"}:
        include_boss = True
    if stage in {"bows02", "bows04", "audit", "all"}:
        include_bows = True

    yielded = set()

    def emit(label, folder):
        key = str(folder).lower()
        if key not in yielded:
            yielded.add(key)
            return (label, Path(folder))
        return None

    if include_boss:
        item = emit("BOSSBase-cover", pipeline.BOSS_COVER)
        if item:
            yield item
        for payload in payloads:
            for algo in ALGOS:
                item = emit(
                    f"BOSSBase-{payload}-{algo}",
                    pipeline.BOSS_STEGO_DIRS[payload][algo],
                )
                if item:
                    yield item

    if include_bows:
        item = emit("BOWS2-cover", pipeline.BOWS2_COVER)
        if item:
            yield item
        for payload in payloads:
            for algo in ALGOS:
                item = emit(
                    f"BOWS2-{payload}-{algo}",
                    pipeline.BOWS2_STEGO_DIRS[payload][algo],
                )
                if item:
                    yield item


def image_integrity_audit(
    pipeline,
    stage: str,
    out_dir: Path,
    limit_per_folder: int,
) -> List[Dict[str, Any]]:
    rows = []
    allowed_exts = pipeline.IMAGE_EXTS

    for label, folder in iter_stage_directories(pipeline, stage):
        folder = Path(folder)
        if not folder.is_dir():
            rows.append({
                "label": label,
                "folder": str(folder),
                "status": "FAIL",
                "reason": "folder_not_found",
                "files_checked": 0,
                "wrong_size": 0,
                "wrong_mode": 0,
                "read_errors": 0,
            })
            continue

        files = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_exts
        )
        if limit_per_folder > 0:
            files = files[:limit_per_folder]

        wrong_size = 0
        wrong_mode = 0
        read_errors = 0
        examples = []

        for path in files:
            try:
                with Image.open(path) as image:
                    if image.size != EXPECTED_IMAGE_SIZE:
                        wrong_size += 1
                        if len(examples) < 5:
                            examples.append(
                                f"{path.name}: size={image.size}"
                            )
                    if image.mode != "L":
                        wrong_mode += 1
                        if len(examples) < 5:
                            examples.append(
                                f"{path.name}: mode={image.mode}"
                            )
            except Exception as exc:
                read_errors += 1
                if len(examples) < 5:
                    examples.append(f"{path.name}: {repr(exc)}")

        passed = (
            len(files) > 0
            and wrong_size == 0
            and wrong_mode == 0
            and read_errors == 0
        )
        row = {
            "label": label,
            "folder": str(folder),
            "files_checked": len(files),
            "wrong_size": wrong_size,
            "wrong_mode": wrong_mode,
            "read_errors": read_errors,
            "examples": " | ".join(examples),
            "status": "PASS" if passed else "FAIL",
        }
        rows.append(row)

        print(
            f"[IMAGE] {label:32s} | checked={len(files):6d} | "
            f"size_errors={wrong_size} | mode_errors={wrong_mode} | "
            f"{row['status']}",
            flush=True,
        )

    save_csv(rows, out_dir / "image_integrity_audit.csv")
    save_json(rows, out_dir / "image_integrity_audit.json")

    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(
            f"Image integrity audit failed for {len(failures)} folder(s). "
            "No bank inference was started."
        )
    return rows


def run_pipeline_stage(
    pipeline,
    stage: str,
    max_images: int,
    batch_size: int,
    tta_count: int,
):
    if stage == "audit":
        return pipeline.audit()
    if stage == "boss02":
        return pipeline.run_internal(
            "0.2bpp",
            max_images,
            batch_size,
            tta_count,
        )
    if stage == "boss04":
        return pipeline.run_internal(
            "0.4bpp",
            max_images,
            batch_size,
            tta_count,
        )
    if stage == "bows04":
        return pipeline.run_external(
            "0.4bpp",
            max_images,
            batch_size,
            tta_count,
        )
    if stage == "bows02":
        return pipeline.run_external(
            "0.2bpp",
            max_images,
            batch_size,
            tta_count,
        )
    if stage == "table":
        return pipeline.make_reviewer_table()
    if stage == "all":
        pipeline.audit()
        pipeline.run_internal(
            "0.2bpp",
            max_images,
            batch_size,
            tta_count,
        )
        pipeline.run_internal(
            "0.4bpp",
            max_images,
            batch_size,
            tta_count,
        )
        pipeline.run_external(
            "0.4bpp",
            max_images,
            batch_size,
            tta_count,
        )
        pipeline.run_external(
            "0.2bpp",
            max_images,
            batch_size,
            tta_count,
        )
        return pipeline.make_reviewer_table()
    raise ValueError(stage)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the five-detector candidate bank with the locked "
            "ACC+ hsnnet architecture and strict no-resize inference."
        )
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "audit",
            "boss02",
            "boss04",
            "bows02",
            "bows04",
            "table",
            "all",
        ],
    )
    parser.add_argument(
        "--pipeline",
        default="",
        help="Path to run_candidate_pipeline_full5.py",
    )
    parser.add_argument(
        "--model-definition",
        default=str(
            Path(__file__).with_name(
                "model_definition_hsnnet_accplus_locked_v2.py"
            )
        ),
    )
    parser.add_argument(
        "--result-folder",
        default="R2_CANDIDATE_RESULTS_HSNNET_LOCKED",
        help=(
            "Fresh result folder under the checkpoint BASE directory. "
            "A new folder prevents reuse of old response caches."
        ),
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta-count", type=int, default=8)
    parser.add_argument(
        "--image-audit-limit",
        type=int,
        default=0,
        help="0 checks every image; positive values check that many per folder.",
    )
    args = parser.parse_args()

    pipeline_path = find_pipeline(args.pipeline or None)
    model_path = Path(args.model_definition).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Locked model definition not found: {model_path}"
        )

    print("=" * 110)
    print("LOCKED HSNNET DETECTOR-BANK RECONSTRUCTION")
    print("Pipeline        :", pipeline_path)
    print("Model definition:", model_path)
    print("Stage           :", args.stage)
    print("TTA count       :", args.tta_count)
    print("Max images      :", args.max_images)
    print("=" * 110)

    pipeline = load_module(
        pipeline_path,
        "candidate_pipeline_locked_runtime",
    )
    model_module = load_module(
        model_path,
        "hsnnet_accplus_locked_runtime",
    )

    # Force the exact model definition and isolate every new output/cache.
    pipeline.REFERENCE_MODEL = model_path
    pipeline.RESULT_ROOT = pipeline.BASE / args.result_folder
    pipeline.RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    # The official bank uses Q10-Q90 and gamma=0.90.
    pipeline.Q_LOW = 10
    pipeline.Q_HIGH = 90
    pipeline.GAMMA = 0.90

    if args.tta_count != 8:
        raise ValueError(
            "The locked official protocol requires exactly 8 D4 TTA elements."
        )

    install_strict_dataset(pipeline)

    audit_out = pipeline.RESULT_ROOT / "00_LOCKED_INTEGRITY_AUDIT"
    audit_out.mkdir(parents=True, exist_ok=True)

    if args.stage in {"boss02", "bows02"}:
        payloads = ["0.2bpp"]
    elif args.stage in {"boss04", "bows04"}:
        payloads = ["0.4bpp"]
    else:
        payloads = PAYLOADS.copy()

    started = time.time()

    architecture_integrity_audit(
        pipeline,
        model_module,
        payloads,
        audit_out,
    )

    if args.stage != "table":
        image_integrity_audit(
            pipeline,
            args.stage,
            audit_out,
            args.image_audit_limit,
        )

    result = run_pipeline_stage(
        pipeline,
        args.stage,
        args.max_images,
        args.batch_size,
        args.tta_count,
    )

    run_manifest = {
        "pipeline": str(pipeline_path),
        "pipeline_sha256": sha256_file(pipeline_path),
        "model_definition": str(model_path),
        "model_definition_sha256": sha256_file(model_path),
        "stage": args.stage,
        "result_root": str(pipeline.RESULT_ROOT),
        "q_low": pipeline.Q_LOW,
        "q_high": pipeline.Q_HIGH,
        "gamma": pipeline.GAMMA,
        "tta_count": args.tta_count,
        "strict_no_resize": True,
        "strict_grayscale": True,
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.time() - started,
        "result": str(result),
    }
    save_json(
        run_manifest,
        pipeline.RESULT_ROOT / "locked_run_manifest.json",
    )

    print("=" * 110)
    print("LOCKED BANK STAGE COMPLETE")
    print("Result root:", pipeline.RESULT_ROOT)
    print("Result     :", result)
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
