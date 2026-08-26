from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

# Torch is imported only for stages that perform inference.
torch = None
nn = None
Dataset = None
DataLoader = None

ALGOS = ["WOW", "S-UNIWARD", "HILL", "HUGO", "MiPOD"]
EXTERNAL_ALGOS = ALGOS.copy()
Q_LOW = 10
Q_HIGH = 90
GAMMA = 0.90
EPS = 1e-9
IMAGE_EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REFERENCE_MODEL = PACKAGE_ROOT / "reference" / "model_definition_accplus.py"


# =============================================================================
# FIXED BOSSBASE DATASET PATHS
# These are the only BOSSBase dataset paths used by this script.
# Dataset paths stored in checkpoint config files are ignored.
# =============================================================================
BOSS_COVER = Path(r"${BOSSBASE_ROOT}\cover")

BOSS_STEGO_DIRS = {
    "0.2bpp": {
        "WOW": Path(r"${BOSSBASE_ROOT}\stego\WOW\0.2bpp\stego"),
        "S-UNIWARD": Path(r"${BOSSBASE_ROOT}\stego\S-UNIWARD\0.2bpp\stego"),
        "HILL": Path(r"${BOSSBASE_ROOT}\stego\HILL\0.2bpp\stego"),
        "HUGO": Path(r"${BOSSBASE_ROOT}\stego\HUGO\0.2bpp\stego"),
        "MiPOD": Path(r"${BOSSBASE_ROOT}\stego\MiPOD\0.2bpp\stego"),
    },
    "0.4bpp": {
        "WOW": Path(r"${BOSSBASE_ROOT}\stego\WOW\0.4bpp\stego"),
        "S-UNIWARD": Path(r"${BOSSBASE_ROOT}\stego\S-UNIWARD\0.4bpp\stego"),
        "HILL": Path(r"${BOSSBASE_ROOT}\stego\HILL\0.4bpp\stego"),
        "HUGO": Path(r"${BOSSBASE_ROOT}\stego\HUGO\0.4bpp\stego"),
        "MiPOD": Path(r"${BOSSBASE_ROOT}\stego\MiPOD\0.4bpp\stego"),
    },
}



# =============================================================================
# FIXED BOWS2 DATASET PATHS — FULL FIVE-ALGORITHM EXTERNAL EVALUATION
# BOWS2 is used only for frozen external testing.
# =============================================================================
BOWS2_ROOT = Path(
    os.environ.get(
        "BOWS2_ROOT",
        str(PACKAGE_ROOT / "data" / "BOWS2"),
    )
).expanduser()

BOWS2_COVER = BOWS2_ROOT / "cover"

BOWS2_STEGO = {
    "0.2bpp": {
        "WOW": BOWS2_ROOT / "stego" / "WOW" / "0.2bpp" / "stego",
        "S-UNIWARD": BOWS2_ROOT / "stego" / "S-UNIWARD" / "0.2bpp" / "stego",
        "HILL": BOWS2_ROOT / "stego" / "HILL" / "0.2bpp" / "stego",
        "HUGO": BOWS2_ROOT / "stego" / "HUGO" / "0.2bpp" / "stego",
        "MiPOD": BOWS2_ROOT / "stego" / "MiPOD" / "0.2bpp" / "stego",
    },
    "0.4bpp": {
        "WOW": BOWS2_ROOT / "stego" / "WOW" / "0.4bpp" / "stego",
        "S-UNIWARD": BOWS2_ROOT / "stego" / "S-UNIWARD" / "0.4bpp" / "stego",
        "HILL": BOWS2_ROOT / "stego" / "HILL" / "0.4bpp" / "stego",
        "HUGO": BOWS2_ROOT / "stego" / "HUGO" / "0.4bpp" / "stego",
        "MiPOD": BOWS2_ROOT / "stego" / "MiPOD" / "0.4bpp" / "stego",
    },
}

def choose_base() -> Path:
    candidates = []

    configured_root = os.environ.get(
        "HSNNET_EXPERIMENT_ROOT"
    )

    if configured_root:
        candidates.append(
            Path(configured_root).expanduser()
        )

    candidates.extend([
        Path.cwd(),
        Path.cwd().parent,
        PACKAGE_ROOT,
        PACKAGE_ROOT.parent,
    ])

    for candidate in candidates:
        try:
            if (
                candidate.is_dir()
                and any(
                    candidate.glob(
                        "outputs_acc_plus_*_only_02bpp"
                    )
                )
            ):
                return candidate.resolve()
        except OSError:
            pass

    raise FileNotFoundError(
        "The original detector-training output folder "
        "was not found. Set HSNNET_EXPERIMENT_ROOT to "
        "the directory containing "
        "outputs_acc_plus_*_only_02bpp."
    )


BASE = choose_base()
RESULT_ROOT = BASE / "R2_POINT4_CANDIDATE_RESULTS"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_algo(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    mapping = {
        "wow": "WOW",
        "s-uniward": "S-UNIWARD",
        "suniward": "S-UNIWARD",
        "hill": "HILL",
        "hugo": "HUGO",
        "mipod": "MiPOD",
    }
    return mapping.get(key, value)


def folder_token(algo: str) -> str:
    return "SUNIWARD" if algo == "S-UNIWARD" else algo


def template_token(algo: str) -> str:
    return algo


def payload_tag(payload: str) -> str:
    return "02bpp" if payload == "0.2bpp" else "04bpp"


def discover_checkpoint(algo: str, payload: str) -> Path:
    tag = payload_tag(payload)
    outer = BASE / f"outputs_acc_plus_{folder_token(algo)}_only_{tag}"
    expected = outer / f"outputs_template_{template_token(algo)}_acc_plus" / "best_model_ema.pth"
    if expected.is_file():
        return expected

    if outer.is_dir():
        hits = sorted(outer.rglob("best_model_ema.pth"))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            preferred = [
                p for p in hits
                if p.parent.name.lower() == f"outputs_template_{template_token(algo)}_acc_plus".lower()
            ]
            if preferred:
                return preferred[0]
            return hits[0]

    # Last controlled fallback under BASE.
    needles = [folder_token(algo).lower(), algo.lower().replace("-", "")]
    hits = []
    for p in BASE.rglob("best_model_ema.pth"):
        low = str(p).lower().replace("-", "")
        if tag.lower() in low and any(n.replace("-", "") in low for n in needles):
            hits.append(p)
    if hits:
        return sorted(hits)[0]

    raise FileNotFoundError(
        f"No best_model_ema.pth found for {algo} {payload}. "
        f"Expected under: {outer}"
    )


def find_run_config(checkpoint: Path) -> Dict[str, Any]:
    folder = checkpoint.parent
    direct = folder / "config.json"
    if direct.is_file():
        return read_json(direct)

    for p in sorted(folder.glob("template*.json")):
        data = read_json(p)
        if isinstance(data, dict):
            if isinstance(data.get("config"), dict):
                return data["config"]
            if isinstance(data.get("current_run"), dict):
                return data["current_run"]
            # Some template files are already flat configs.
            if "cover_dir" in data or "image_size" in data:
                return data

    raise FileNotFoundError(
        f"No config.json or usable template*.json beside checkpoint: {checkpoint}"
    )


def list_images(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted(
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def first_existing_dir(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p.is_dir():
                return p.resolve()
        except OSError:
            continue
    return None


def resolve_boss_dirs(
    algo: str,
    payload: str,
    run_cfg: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Return only the fixed BOSSBase paths declared in BOSS_COVER and
    BOSS_STEGO_DIRS. Dataset paths from run_cfg are deliberately ignored.
    """
    if payload not in BOSS_STEGO_DIRS:
        raise ValueError(f"Unsupported payload: {payload}")
    if algo not in BOSS_STEGO_DIRS[payload]:
        raise ValueError(f"Unsupported algorithm: {algo}")

    cover = BOSS_COVER
    stego = BOSS_STEGO_DIRS[payload][algo]

    if not cover.is_dir():
        raise FileNotFoundError(f"Fixed cover directory does not exist: {cover}")
    if not stego.is_dir():
        raise FileNotFoundError(
            f"Fixed stego directory does not exist for {algo} {payload}: {stego}"
        )
    return cover, stego

def resolve_bows2_root() -> Path:
    root = BOWS2_COVER.parent
    if not root.is_dir():
        raise FileNotFoundError(f"Fixed BOWS2 root does not exist: {root}")
    return root


def resolve_bows2_cover(root: Path) -> Path:
    del root
    if not BOWS2_COVER.is_dir():
        raise FileNotFoundError(
            f"Fixed BOWS2 cover directory does not exist: {BOWS2_COVER}"
        )
    return BOWS2_COVER


def resolve_bows2_stego(root: Path, algo: str, payload: str) -> Path:
    del root
    if payload not in BOWS2_STEGO_DIRS:
        raise ValueError(f"Unsupported BOWS2 payload: {payload}")
    if algo not in BOWS2_STEGO_DIRS[payload]:
        raise ValueError(f"Unsupported BOWS2 algorithm: {algo}")

    folder = BOWS2_STEGO_DIRS[payload][algo]
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Fixed BOWS2 stego directory does not exist for "
            f"{algo} {payload}: {folder}"
        )
    return folder

def split_ids(
    names: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    items = list(names)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * float(train_ratio))
    n_val = int(n * float(val_ratio))
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]


def build_manifest(payload: str) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {}
    val_sets = []
    test_sets = []

    for algo in ALGOS:
        ckpt = discover_checkpoint(algo, payload)
        cfg = find_run_config(ckpt)
        cover, stego = resolve_boss_dirs(algo, payload, cfg)

        matched = sorted(set(list_images(cover)).intersection(list_images(stego)))
        if not matched:
            raise RuntimeError(f"No matched BOSSBase images for {algo} {payload}")

        train_ratio = float(cfg.get("train_ratio", 0.70))
        val_ratio = float(cfg.get("val_ratio", 0.15))
        seed = int(cfg.get("split_seed", cfg.get("seed", 42)))
        train_ids, val_ids, test_ids = split_ids(
            matched, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

        val_sets.append(set(val_ids))
        test_sets.append(set(test_ids))

        manifest[algo] = {
            "checkpoint": str(ckpt),
            "template_dir": str(ckpt.parent),
            "config_file": str(
                ckpt.parent / "config.json"
                if (ckpt.parent / "config.json").is_file()
                else ""
            ),
            "cover_dir": str(cover),
            "stego_dir": str(stego),
            "image_size": int(cfg.get("image_size", 256)),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "split_seed": seed,
            "matched_pairs": len(matched),
            "train_pairs": len(train_ids),
            "val_pairs": len(val_ids),
            "test_pairs": len(test_ids),
            "run_config": cfg,
        }

    common_val = sorted(set.intersection(*val_sets))
    common_test = sorted(set.intersection(*test_sets))
    if not common_val or not common_test:
        raise RuntimeError(
            f"No common leakage-safe split across all five models at {payload}."
        )

    manifest["_shared"] = {
        "payload": payload,
        "common_val_ids": common_val,
        "common_test_ids": common_test,
        "common_val_pairs": len(common_val),
        "common_test_pairs": len(common_test),
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "gamma": GAMMA,
    }
    return manifest


def import_torch() -> None:
    global torch, nn, Dataset, DataLoader
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
    torch, nn, Dataset, DataLoader = _torch, _nn, _Dataset, _DataLoader


def load_model_module():
    script_candidates = [
        REFERENCE_MODEL,
        BASE / "model_definition_accplus.py",
        BASE / "train_hsnnet_acc_plus_all_algorithms.py",
        BASE / "train_hsnnet_acc_plus_all_algorithms_02bpp.py",
        BASE.parent / "H37-ALL" / "train_hsnnet_acc_plus_all_algorithms.py",
        BASE.parent / "H37-ALL" / "train_hsnnet_acc_plus_all_algorithms_02bpp.py",
    ]
    script = next((p for p in script_candidates if p.is_file()), None)
    if script is None:
        raise FileNotFoundError(
            "Could not find the ACC+ model-definition script. "
            f"Bundled reference expected at {REFERENCE_MODEL}"
        )
    spec = importlib.util.spec_from_file_location("accplus_runtime", str(script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script


def build_cfg_obj(module, algo: str, payload: str, cfg_dict: Dict[str, Any]):
    cfg = module.make_cfg_for_algorithm(algo)
    for key, value in cfg_dict.items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    cfg.algorithm = algo
    cfg.payload = payload
    cfg.image_size = int(cfg_dict.get("image_size", 256))
    cfg.use_resize = bool(cfg_dict.get("use_resize", True))
    cfg.use_amp = bool(cfg_dict.get("use_amp", True))
    cfg.use_final_tta = True
    cfg.tta_d4_count = int(cfg_dict.get("tta_d4_count", 8))
    return cfg


def clean_state_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        for key in ["state_dict", "model_state_dict", "ema_state_dict", "model", "net"]:
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise TypeError("Unsupported checkpoint format")
    cleaned = {}
    for key, value in raw.items():
        new_key = key
        for prefix in ["module.", "model.", "ema."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def resolve_runtime_path(value: Any) -> Path:
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()

    if "${" in expanded:
        raise RuntimeError(
            "An unresolved path placeholder remains: "
            f"{expanded}. Set the corresponding "
            "environment variable before raw-image inference."
        )

    if not path.is_absolute():
        path = PACKAGE_ROOT / path

    return path.resolve()


def load_model(module, algo: str, payload: str, meta: Dict[str, Any], device):
    cfg = build_cfg_obj(module, algo, payload, meta["run_config"])
    model = module.hsnnet(cfg).to(device)
    raw = torch.load(resolve_runtime_path(meta["checkpoint"]), map_location=device)
    state = clean_state_dict(raw)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


class ResponseDataset:
    pass


def make_dataset_class():
    class _ResponseDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]], image_size: int):
            self.rows = rows
            self.image_size = int(image_size)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(row["path"]).convert("L")
            if image.size != (self.image_size, self.image_size):
                try:
                    bilinear = Image.Resampling.BILINEAR
                except AttributeError:
                    bilinear = Image.BILINEAR
                image = image.resize((self.image_size, self.image_size), bilinear)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0)
            return x, int(index)
    return _ResponseDataset


def d4_batch(module, x, g: int):
    if hasattr(module, "_apply_d4_batch"):
        return module._apply_d4_batch(x, g)
    if g == 0: return x
    if g == 1: return torch.rot90(x, 1, [2, 3])
    if g == 2: return torch.rot90(x, 2, [2, 3])
    if g == 3: return torch.rot90(x, 3, [2, 3])
    if g == 4: return torch.flip(x, [3])
    if g == 5: return torch.flip(x, [2])
    if g == 6: return torch.rot90(torch.flip(x, [3]), 1, [2, 3])
    if g == 7: return torch.rot90(torch.flip(x, [2]), 1, [2, 3])
    raise ValueError(g)


def autocast_context(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


@torch.no_grad() if False else (lambda f: f)
def placeholder():
    pass


def infer_detector(
    module,
    model,
    cfg,
    rows: List[Dict[str, Any]],
    batch_size: int,
    tta_count: int,
    device,
) -> np.ndarray:
    DatasetClass = make_dataset_class()
    dataset = DatasetClass(rows, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    output = np.empty(len(rows), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        done = 0
        for x, indices in loader:
            x = x.to(device, non_blocking=True)
            logits_sum = None
            for g in range(int(tta_count)):
                xg = d4_batch(module, x, g)
                with autocast_context(device, bool(cfg.use_amp)):
                    logits = model(xg)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            logits = logits_sum / float(tta_count)
            probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy().astype(np.float32)
            idx = indices.numpy().astype(int)
            output[idx] = probs
            done += len(idx)
            print(f"    {done}/{len(rows)}", flush=True)
    return output


def make_internal_rows(manifest: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    shared = manifest["_shared"]
    ids = shared["common_val_ids"] if split == "validation" else shared["common_test_ids"]
    ref = manifest[ALGOS[0]]
    rows = []

    cover_dir = resolve_runtime_path(ref["cover_dir"])
    for name in ids:
        rows.append({
            "sample_id": f"COVER__{split}__{name}",
            "image_name": name,
            "true_algo": "COVER",
            "is_stego": 0,
            "path": str(cover_dir / name),
        })

    for algo in ALGOS:
        stego_dir = resolve_runtime_path(manifest[algo]["stego_dir"])
        for name in ids:
            rows.append({
                "sample_id": f"{algo}__{split}__{name}",
                "image_name": name,
                "true_algo": algo,
                "is_stego": 1,
                "path": str(stego_dir / name),
            })
    return rows


def run_response_matrix(
    payload: str,
    rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    cache_dir: Path,
    output_csv: Path,
    batch_size: int,
    tta_count: int,
) -> pd.DataFrame:
    import_torch()
    module, script = load_model_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model script:", script)
    print("Device:", device)

    base_df = pd.DataFrame(rows)
    response = base_df[["sample_id", "image_name", "true_algo", "is_stego", "path"]].copy()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for detector in ALGOS:
        cache_path = cache_dir / f"p_{folder_token(detector)}.npy"
        if cache_path.is_file():
            cached = np.load(cache_path)
            if len(cached) == len(rows):
                print(f"[CACHE] {detector}: {cache_path}")
                response[f"p_{detector}"] = cached
                continue

        print("=" * 100)
        print(f"INFERENCE | payload={payload} | detector={detector} | samples={len(rows)}")
        print("Checkpoint:", manifest[detector]["checkpoint"])
        model, cfg = load_model(module, detector, payload, manifest[detector], device)
        probs = infer_detector(
            module=module,
            model=model,
            cfg=cfg,
            rows=rows,
            batch_size=batch_size,
            tta_count=tta_count,
            device=device,
        )
        np.save(cache_path, probs.astype(np.float32))
        response[f"p_{detector}"] = probs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    response.to_csv(output_csv, index=False)
    return response


def calibrate_z(
    validation: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
    cover = validation[validation["is_stego"] == 0]
    calibration = {}
    output_frames = [frame.copy() for frame in frames]

    for detector in ALGOS:
        mu = float(cover[f"p_{detector}"].mean())
        std = float(cover[f"p_{detector}"].std(ddof=0) + EPS)
        calibration[detector] = {"mu_cover": mu, "std_cover": std}
        for frame in output_frames:
            frame[f"z_{detector}"] = (frame[f"p_{detector}"] - mu) / std
    return calibration, output_frames


def distance_to_range(z: float, low: float, high: float) -> float:
    if low <= z <= high:
        return 0.0
    if z < low:
        return low - z
    return z - high


def soft_range_score(z: float, low: float, high: float) -> float:
    return float(np.exp(-distance_to_range(z, low, high)))


def build_signatures(
    validation_z: pd.DataFrame,
    q_low: int = Q_LOW,
    q_high: int = Q_HIGH,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    signatures: Dict[str, Any] = {}
    table_rows = []

    for target in ALGOS:
        subset = validation_z[
            (validation_z["is_stego"] == 1) &
            (validation_z["true_algo"] == target)
        ]
        if subset.empty:
            raise RuntimeError(f"No validation stego samples for {target}")

        signatures[target] = {}
        for detector in ALGOS:
            values = subset[f"z_{detector}"].to_numpy(dtype=np.float64)
            stats = {
                "q05": float(np.percentile(values, 5)),
                "q10": float(np.percentile(values, 10)),
                "q25": float(np.percentile(values, 25)),
                "q50": float(np.percentile(values, 50)),
                "q75": float(np.percentile(values, 75)),
                "q90": float(np.percentile(values, 90)),
                "q95": float(np.percentile(values, 95)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values) + EPS),
                "range_low": float(np.percentile(values, q_low)),
                "range_high": float(np.percentile(values, q_high)),
            }
            signatures[target][detector] = stats
            table_rows.append({
                "target_algo": target,
                "template_dimension": detector,
                "q_low_used": q_low,
                "q_high_used": q_high,
                **stats,
            })
    return signatures, pd.DataFrame(table_rows)


def apply_candidate(
    frame_z: pd.DataFrame,
    signatures: Dict[str, Any],
    gamma: float = GAMMA,
) -> pd.DataFrame:
    rows = []
    for _, row in frame_z.iterrows():
        scores = {}
        hard_scores = {}
        inside_counts = {}

        for target in ALGOS:
            soft = []
            hard = []
            for detector in ALGOS:
                z = float(row[f"z_{detector}"])
                low = float(signatures[target][detector]["range_low"])
                high = float(signatures[target][detector]["range_high"])
                soft.append(soft_range_score(z, low, high))
                hard.append(int(low <= z <= high))
            scores[target] = float(np.mean(soft))
            hard_scores[target] = float(np.mean(hard))
            inside_counts[target] = int(np.sum(hard))

        ranked = sorted(ALGOS, key=lambda a: scores[a], reverse=True)
        candidates = [a for a in ranked if scores[a] >= gamma]

        if len(candidates) == 0:
            decision = "UNKNOWN_OR_COVER"
            decision_top1 = "UNKNOWN"
            decision_top2 = ""
        elif len(candidates) == 1:
            decision = "SINGLE_CANDIDATE"
            decision_top1 = candidates[0]
            decision_top2 = ""
        else:
            decision = "MULTI_CANDIDATE"
            decision_top1 = ranked[0]
            decision_top2 = ranked[1]

        true_algo = str(row["true_algo"])
        true_rank = ranked.index(true_algo) + 1 if true_algo in ALGOS else None

        output = {
            "sample_id": row.get("sample_id", ""),
            "image_name": row.get("image_name", ""),
            "true_algo": true_algo,
            "is_stego": int(row["is_stego"]),
            "decision": decision,
            "top1": decision_top1,
            "top2": decision_top2,
            "rank1_full": ranked[0],
            "rank2_full": ranked[1],
            "rank3_full": ranked[2],
            "rank4_full": ranked[3],
            "rank5_full": ranked[4],
            "true_rank": true_rank,
            "candidate_set": ",".join(candidates),
            "candidate_size": len(candidates),
            "candidate_contains_true": int(true_algo in candidates) if true_algo in ALGOS else 0,
            "top1_score": scores[ranked[0]],
            "top2_score": scores[ranked[1]],
            "score_gap": scores[ranked[0]] - scores[ranked[1]],
        }
        for algo in ALGOS:
            output[f"score_{algo}"] = scores[algo]
            output[f"hard_{algo}"] = hard_scores[algo]
            output[f"inside_count_{algo}"] = inside_counts[algo]
        rows.append(output)

    return pd.DataFrame(rows)


def summarize_predictions(
    predictions: pd.DataFrame,
    true_algos: Sequence[str],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    stego = predictions[
        (predictions["is_stego"] == 1) &
        predictions["true_algo"].isin(true_algos)
    ].copy()
    cover = predictions[predictions["is_stego"] == 0].copy()

    summary = {
        "num_total": int(len(predictions)),
        "num_stego": int(len(stego)),
        "num_cover": int(len(cover)),
        "ranked_top1_accuracy": float((stego["true_rank"] <= 1).mean()),
        "ranked_top2_accuracy": float((stego["true_rank"] <= 2).mean()),
        "ranked_top3_accuracy": float((stego["true_rank"] <= 3).mean()),
        "decision_top1_accuracy": float((stego["top1"] == stego["true_algo"]).mean()),
        "candidate_set_accuracy": float(stego["candidate_contains_true"].mean()),
        "stego_unknown_rate": float((stego["candidate_size"] == 0).mean()),
        "stego_single_candidate_rate": float((stego["candidate_size"] == 1).mean()),
        "stego_multi_candidate_rate": float((stego["candidate_size"] > 1).mean()),
        "stego_mean_candidate_size": float(stego["candidate_size"].mean()),
        "mean_true_rank": float(stego["true_rank"].mean()),
        "cover_unknown_rate": float((cover["candidate_size"] == 0).mean()) if len(cover) else None,
        "cover_false_attribution_rate": float((cover["candidate_size"] > 0).mean()) if len(cover) else None,
        "cover_mean_candidate_size": float(cover["candidate_size"].mean()) if len(cover) else None,
    }

    per_algo_rows = []
    for algo in true_algos:
        subset = stego[stego["true_algo"] == algo]
        if subset.empty:
            continue
        per_algo_rows.append({
            "algorithm": algo,
            "n": int(len(subset)),
            "ranked_top1": float((subset["true_rank"] <= 1).mean()),
            "ranked_top2": float((subset["true_rank"] <= 2).mean()),
            "ranked_top3": float((subset["true_rank"] <= 3).mean()),
            "decision_top1": float((subset["top1"] == subset["true_algo"]).mean()),
            "candidate_retention": float(subset["candidate_contains_true"].mean()),
            "unknown_rate": float((subset["candidate_size"] == 0).mean()),
            "single_rate": float((subset["candidate_size"] == 1).mean()),
            "multi_rate": float((subset["candidate_size"] > 1).mean()),
            "mean_candidate_size": float(subset["candidate_size"].mean()),
            "mean_true_rank": float(subset["true_rank"].mean()),
        })
    per_algo = pd.DataFrame(per_algo_rows)

    labels = list(ALGOS) + ["UNKNOWN"]
    cm = confusion_matrix(stego["true_algo"], stego["top1"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return summary, per_algo, cm_df


def bootstrap_ci(
    predictions: pd.DataFrame,
    true_algos: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    stego = predictions[
        (predictions["is_stego"] == 1) &
        predictions["true_algo"].isin(true_algos)
    ].reset_index(drop=True)
    if stego.empty:
        return {}

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(stego), len(stego))
        sample = stego.iloc[idx]
        values.append([
            float((sample["true_rank"] <= 1).mean()),
            float((sample["true_rank"] <= 3).mean()),
            float(sample["candidate_contains_true"].mean()),
            float(sample["true_rank"].mean()),
            float(sample["candidate_size"].mean()),
        ])
    arr = np.asarray(values, dtype=np.float64)
    names = [
        "ranked_top1",
        "ranked_top3",
        "candidate_retention",
        "mean_true_rank",
        "mean_candidate_size",
    ]
    return {
        name: {
            "low": float(np.quantile(arr[:, i], 0.025)),
            "high": float(np.quantile(arr[:, i], 0.975)),
        }
        for i, name in enumerate(names)
    }


def run_internal(
    payload: str,
    max_images: int,
    batch_size: int,
    tta_count: int,
) -> Path:
    tag = payload_tag(payload)
    out = RESULT_ROOT / f"BOSSBASE_CANDIDATE_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(payload)
    if max_images > 0:
        manifest["_shared"]["common_val_ids"] = manifest["_shared"]["common_val_ids"][:max_images]
        manifest["_shared"]["common_test_ids"] = manifest["_shared"]["common_test_ids"][:max_images]
        manifest["_shared"]["common_val_pairs"] = len(manifest["_shared"]["common_val_ids"])
        manifest["_shared"]["common_test_pairs"] = len(manifest["_shared"]["common_test_ids"])
        manifest["_shared"]["smoke_test_only"] = True
    save_json(manifest, out / "template_manifest.json")

    validation_rows = make_internal_rows(manifest, "validation")
    test_rows = make_internal_rows(manifest, "test")

    validation = run_response_matrix(
        payload,
        validation_rows,
        manifest,
        cache_dir=out / "cache_validation",
        output_csv=out / "validation_response_matrix_raw.csv",
        batch_size=batch_size,
        tta_count=tta_count,
    )
    test = run_response_matrix(
        payload,
        test_rows,
        manifest,
        cache_dir=out / "cache_test",
        output_csv=out / "test_response_matrix_raw.csv",
        batch_size=batch_size,
        tta_count=tta_count,
    )

    calibration, [validation_z, test_z] = calibrate_z(validation, [validation, test])
    save_json(calibration, out / "dimension_cover_calibration.json")
    validation_z.to_csv(out / "validation_response_matrix_z.csv", index=False)
    test_z.to_csv(out / "test_response_matrix_z.csv", index=False)

    signatures, signature_table = build_signatures(validation_z, Q_LOW, Q_HIGH)
    save_json(signatures, out / "final_cross_template_signatures.json")
    signature_table.to_csv(out / "final_cross_template_signature_table.csv", index=False)

    predictions = apply_candidate(test_z, signatures, GAMMA)
    predictions.to_csv(out / "final_predictions_all.csv", index=False)
    predictions[predictions["is_stego"] == 1].to_csv(
        out / "final_predictions_stego.csv", index=False
    )
    predictions[predictions["is_stego"] == 0].to_csv(
        out / "final_predictions_cover.csv", index=False
    )

    summary, per_algo, cm = summarize_predictions(predictions, ALGOS)
    summary.update({
        "method": "cross_template_response_signature_candidate_attribution",
        "dataset": "BOSSBase 1.01",
        "payload": payload,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "gamma": GAMMA,
        "candidate_universe": ALGOS,
        "tta_count": tta_count,
        "bootstrap_95ci": bootstrap_ci(predictions, ALGOS),
    })
    save_json(summary, out / "final_summary.json")
    per_algo.to_csv(out / "final_per_algorithm.csv", index=False)
    cm.to_csv(out / "final_top1_confusion_matrix.csv")

    print("=" * 100)
    print(f"COMPLETED INTERNAL CANDIDATE BANK | {payload}")
    print(json.dumps(summary, indent=2))
    print("Saved:", out)
    print("=" * 100)
    return out


def load_bank(payload: str) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    tag = payload_tag(payload)
    bank = RESULT_ROOT / f"BOSSBASE_CANDIDATE_{tag}"
    calibration_path = bank / "dimension_cover_calibration.json"
    signatures_path = bank / "final_cross_template_signatures.json"
    manifest_path = bank / "template_manifest.json"

    for p in [calibration_path, signatures_path, manifest_path]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing bank artifact: {p}\nRun internal stage for {payload} first."
            )
    return read_json(calibration_path), read_json(signatures_path), manifest_path


def make_external_rows(
    payload: str,
    max_images: int,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, str]]:
    root = resolve_bows2_root()
    cover = resolve_bows2_cover(root)
    stego_dirs = {
        algo: resolve_bows2_stego(root, algo, payload)
        for algo in EXTERNAL_ALGOS
    }

    common = set(list_images(cover))
    for folder in stego_dirs.values():
        common &= set(list_images(folder))
    names = sorted(common)
    if max_images > 0:
        names = names[:max_images]
    if not names:
        raise RuntimeError(f"No common BOWS2 names for {payload}")

    rows = []
    for name in names:
        rows.append({
            "sample_id": f"COVER__BOWS2__{payload}__{name}",
            "image_name": name,
            "true_algo": "COVER",
            "is_stego": 0,
            "path": str(cover / name),
        })
    for algo in EXTERNAL_ALGOS:
        for name in names:
            rows.append({
                "sample_id": f"{algo}__BOWS2__{payload}__{name}",
                "image_name": name,
                "true_algo": algo,
                "is_stego": 1,
                "path": str(stego_dirs[algo] / name),
            })
    paths = {
        "bows2_root": str(root),
        "cover_dir": str(cover),
        **{f"{a}_stego_dir": str(p) for a, p in stego_dirs.items()},
    }
    return rows, len(names), paths


def run_external(
    payload: str,
    max_images: int,
    batch_size: int,
    tta_count: int,
) -> Path:
    tag = payload_tag(payload)
    out = RESULT_ROOT / f"BOWS2_EXTERNAL_FULL5_CANDIDATE_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    calibration, signatures, manifest_path = load_bank(payload)
    manifest = read_json(manifest_path)
    rows, n_per_class, external_paths = make_external_rows(payload, max_images)

    response = run_response_matrix(
        payload,
        rows,
        manifest,
        cache_dir=out / "cache",
        output_csv=out / "external_response_matrix_raw.csv",
        batch_size=batch_size,
        tta_count=tta_count,
    )

    response_z = response.copy()
    for detector in ALGOS:
        mu = float(calibration[detector]["mu_cover"])
        std = float(calibration[detector]["std_cover"])
        response_z[f"z_{detector}"] = (
            response_z[f"p_{detector}"] - mu
        ) / (std + EPS)
    response_z.to_csv(out / "external_response_matrix_z.csv", index=False)

    predictions = apply_candidate(response_z, signatures, GAMMA)
    predictions.to_csv(out / "external_candidate_predictions.csv", index=False)

    summary, per_algo, cm = summarize_predictions(predictions, EXTERNAL_ALGOS)
    summary.update({
        "method": "full_five_way_frozen_external_cross_template_candidate_attribution",
        "dataset": "BOWS2",
        "payload": payload,
        "external_true_algorithms": EXTERNAL_ALGOS,
        "external_true_class_count": len(EXTERNAL_ALGOS),
        "candidate_universe_count": len(ALGOS),
        "evaluation_scope": "complete five-algorithm external attribution",
        "candidate_universe": ALGOS,
        "matched_images_per_class": n_per_class,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "gamma": GAMMA,
        "tta_count": tta_count,
        "protocol": (
            "BOSSBase-derived frozen calibration and signatures; "
            "complete five-algorithm BOWS2 true-class evaluation; "
            "no BOWS2 training, calibration, signature construction, "
            "threshold selection, Q-range selection, or gamma tuning."
        ),
        "paths": external_paths,
        "bootstrap_95ci": bootstrap_ci(predictions, EXTERNAL_ALGOS),
    })
    save_json(summary, out / "external_candidate_summary.json")
    per_algo.to_csv(out / "external_candidate_per_algorithm.csv", index=False)
    cm.to_csv(out / "external_top1_confusion_matrix.csv")

    print("=" * 100)
    print(f"COMPLETED FULL FIVE-WAY EXTERNAL CANDIDATE TEST | BOWS2 | {payload}")
    print(json.dumps(summary, indent=2))
    print("Saved:", out)
    print("=" * 100)
    return out


def audit() -> Path:
    out = RESULT_ROOT / "00_AUDIT_FULL5"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for payload in ["0.2bpp", "0.4bpp"]:
        for algo in ALGOS:
            try:
                ckpt = discover_checkpoint(algo, payload)
                cfg = find_run_config(ckpt)
                cover, stego = resolve_boss_dirs(algo, payload, cfg)
                matched = len(set(list_images(cover)).intersection(list_images(stego)))
                status = "READY"
                error = ""
            except Exception as exc:
                ckpt = Path()
                cover = Path()
                stego = Path()
                matched = 0
                status = "ERROR"
                error = repr(exc)

            rows.append({
                "payload": payload,
                "algorithm": algo,
                "status": status,
                "checkpoint": str(ckpt) if str(ckpt) != "." else "",
                "cover_dir": str(cover) if str(cover) != "." else "",
                "stego_dir": str(stego) if str(stego) != "." else "",
                "matched_pairs": matched,
                "error": error,
            })

    try:
        bows_root = resolve_bows2_root()
        bows_cover = resolve_bows2_cover(bows_root)
        cover_names = set(list_images(bows_cover))
        for payload in ["0.2bpp", "0.4bpp"]:
            for algo in EXTERNAL_ALGOS:
                try:
                    folder = resolve_bows2_stego(bows_root, algo, payload)
                    matched = len(cover_names.intersection(list_images(folder)))
                    status = "READY"
                    error = ""
                except Exception as exc:
                    folder = Path()
                    matched = 0
                    status = "ERROR"
                    error = repr(exc)
                rows.append({
                    "payload": payload,
                    "algorithm": f"BOWS2-{algo}",
                    "status": status,
                    "checkpoint": "",
                    "cover_dir": str(bows_cover),
                    "stego_dir": str(folder) if str(folder) != "." else "",
                    "matched_pairs": matched,
                    "error": error,
                })
    except Exception as exc:
        rows.append({
            "payload": "both",
            "algorithm": "BOWS2",
            "status": "ERROR",
            "checkpoint": "",
            "cover_dir": "",
            "stego_dir": "",
            "matched_pairs": 0,
            "error": repr(exc),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "audit_report.csv", index=False)
    save_json(rows, out / "audit_report.json")

    print("=" * 130)
    print("42-exbow2 FULL FIVE-WAY CANDIDATE AUDIT")
    print("BASE:", BASE)
    print("=" * 130)
    print(df.to_string(index=False))
    print("=" * 130)
    print("Saved:", out / "audit_report.csv")
    return out


def make_reviewer_table() -> Path:
    out = RESULT_ROOT / "05_REVIEWER_TABLE_FULL5"
    out.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "evaluation": "BOSSBase internal candidate bank",
            "source_condition": "Matched source",
            "payload": "0.2bpp",
            "path": RESULT_ROOT / "BOSSBASE_CANDIDATE_02bpp" / "final_summary.json",
            "interpretation": "Complete five-algorithm matched-source low-payload evaluation.",
        },
        {
            "evaluation": "BOSSBase internal candidate bank",
            "source_condition": "Matched source",
            "payload": "0.4bpp",
            "path": RESULT_ROOT / "BOSSBASE_CANDIDATE_04bpp" / "final_summary.json",
            "interpretation": "Complete five-algorithm matched-source main-payload evaluation.",
        },
        {
            "evaluation": "BOWS2 frozen external candidate bank",
            "source_condition": "External source",
            "payload": "0.4bpp",
            "path": RESULT_ROOT / "BOWS2_EXTERNAL_FULL5_CANDIDATE_04bpp" / "external_candidate_summary.json",
            "interpretation": "Complete five-algorithm frozen external evaluation; no BOWS2 adaptation.",
        },
        {
            "evaluation": "BOWS2 frozen external candidate bank",
            "source_condition": "External source",
            "payload": "0.2bpp",
            "path": RESULT_ROOT / "BOWS2_EXTERNAL_FULL5_CANDIDATE_02bpp" / "external_candidate_summary.json",
            "interpretation": "Complete five-algorithm frozen external low-payload evaluation; no BOWS2 adaptation.",
        },
    ]

    rows = []
    input_files = {}
    all_complete = True

    for exp in experiments:
        path = exp["path"]
        input_files[f'{exp["evaluation"]} | {exp["payload"]}'] = str(path)
        data = read_json(path) if path.is_file() else None
        if data is None:
            all_complete = False

        rows.append({
            "Evaluation": exp["evaluation"],
            "Source condition": exp["source_condition"],
            "Payload": exp["payload"],
            "True classes": "5",
            "Candidate universe": "5",
            "Top-1": f'{100*data["ranked_top1_accuracy"]:.2f}%' if data else "PENDING",
            "Top-2": f'{100*data["ranked_top2_accuracy"]:.2f}%' if data else "PENDING",
            "Top-3": f'{100*data["ranked_top3_accuracy"]:.2f}%' if data else "PENDING",
            "Candidate retention": (
                f'{100*data["candidate_set_accuracy"]:.2f}%' if data else "PENDING"
            ),
            "Mean true rank": f'{data["mean_true_rank"]:.3f}' if data else "PENDING",
            "Mean set size": (
                f'{data["stego_mean_candidate_size"]:.3f}' if data else "PENDING"
            ),
            "Stego unknown rate": (
                f'{100*data["stego_unknown_rate"]:.2f}%' if data else "PENDING"
            ),
            "Cover rejection": (
                f'{100*data["cover_unknown_rate"]:.2f}%'
                if data and data.get("cover_unknown_rate") is not None
                else "PENDING"
            ),
            "Status": "COMPLETE" if data else "PENDING",
            "Interpretation": exp["interpretation"],
        })

    table = pd.DataFrame(rows)
    csv_path = out / "reviewer2_point4_full5_candidate_evidence.csv"
    md_path = out / "reviewer2_point4_full5_candidate_evidence.md"
    json_path = out / "reviewer2_point4_full5_completion_status.json"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    headers = list(table.columns)
    md = [
        "# Reviewer 2 — Point 4: Full Five-Algorithm Candidate-Bank Evidence",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in table.iterrows():
        md.append(
            "| " + " | ".join(
                str(row[h]).replace("|", "/").replace("\n", " ")
                for h in headers
            ) + " |"
        )

    md.extend([
        "",
        "Both datasets are evaluated with five true embedding algorithms and "
        "the same five-algorithm candidate universe. All evaluations use the "
        "locked Q10–Q90 response-signature ranges and gamma = 0.90. BOWS2 is "
        "used only for frozen external testing with BOSSBase-derived detectors, "
        "calibration, and signatures; no BOWS2 training or tuning is performed.",
    ])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    status = {
        "root": str(RESULT_ROOT),
        "all_complete": all_complete,
        "fully_matched_five_way_task": EXTERNAL_ALGOS == ALGOS,
        "BOSSBase_true_algorithms": ALGOS,
        "BOWS2_true_algorithms": EXTERNAL_ALGOS,
        "candidate_universe": ALGOS,
        "input_files": input_files,
        "outputs": {
            "csv": str(csv_path),
            "markdown": str(md_path),
            "json": str(json_path),
        },
        "rows": rows,
    }
    save_json(status, json_path)

    print("=" * 140)
    print("REVIEWER 2 POINT 4 — FULL FIVE-WAY EVIDENCE")
    print("all_complete:", all_complete)
    print(table.to_string(index=False))
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("Saved:", json_path)
    print("=" * 140)
    return out

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["audit", "boss02", "boss04", "bows02", "bows04", "table", "all"],
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 = official full run; positive value = smoke test per class.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta-count", type=int, default=8)
    args = parser.parse_args()

    print("=" * 100)
    print("R2 POINT 4 — FULL FIVE-WAY CANDIDATE PIPELINE")
    print("BASE       :", BASE)
    print("RESULT_ROOT:", RESULT_ROOT)
    print("STAGE      :", args.stage)
    print("MAX_IMAGES :", args.max_images)
    print("BOSS COVER :", BOSS_COVER)
    if args.stage in {"boss02", "bows02"}:
        print("BOSS STEGO 0.2:")
        for _algo in ALGOS:
            print(f"  {_algo:11s}: {BOSS_STEGO_DIRS['0.2bpp'][_algo]}")
    if args.stage in {"boss04", "bows04"}:
        print("BOSS STEGO 0.4:")
        for _algo in ALGOS:
            print(f"  {_algo:11s}: {BOSS_STEGO_DIRS['0.4bpp'][_algo]}")
    if args.stage in {"bows02", "bows04", "all"}:
        print("BOWS2 COVER:", BOWS2_COVER)
        _payloads = (
            ["0.2bpp", "0.4bpp"] if args.stage == "all"
            else ["0.2bpp" if args.stage == "bows02" else "0.4bpp"]
        )
        for _payload in _payloads:
            print(f"BOWS2 STEGO {_payload}:")
            for _algo in EXTERNAL_ALGOS:
                print(f"  {_algo:11s}: {BOWS2_STEGO_DIRS[_payload][_algo]}")
    print("=" * 100)

    if args.stage in {"audit", "all"}:
        audit()
    if args.stage in {"boss02", "all"}:
        run_internal("0.2bpp", args.max_images, args.batch_size, args.tta_count)
    if args.stage in {"boss04", "all"}:
        run_internal("0.4bpp", args.max_images, args.batch_size, args.tta_count)
    if args.stage in {"bows04", "all"}:
        run_external("0.4bpp", args.max_images, args.batch_size, args.tta_count)
    if args.stage in {"bows02", "all"}:
        run_external("0.2bpp", args.max_images, args.batch_size, args.tta_count)
    if args.stage in {"table", "all"}:
        make_reviewer_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
