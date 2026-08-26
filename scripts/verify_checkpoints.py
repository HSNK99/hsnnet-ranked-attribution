from __future__ import annotations
import hashlib, importlib.util, sys
from pathlib import Path
from typing import Any
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
MODEL_FILE=ROOT/"src"/"model_definition_hsnnet_locked.py"
INVENTORY_FILE=ROOT/"manifests"/"checkpoint_inventory.csv"
OUT_DIR=ROOT/"generated"
OUT_DIR.mkdir(parents=True,exist_ok=True)

EXPECTED_TOTAL=7_293_009
EXPECTED_TRAINABLE=7_292_609
EXPECTED_FRONTEND=400

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def import_module(path):
    spec=importlib.util.spec_from_file_location("hsnnet_locked_model",str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def tensor_dict_candidates(value:Any,label="root"):
    out=[]
    if isinstance(value,dict):
        tensors={str(k):v for k,v in value.items() if torch.is_tensor(v)}
        if len(tensors)>=20:
            out.append((label,tensors))
        for key,item in value.items():
            if isinstance(item,dict):
                out.extend(tensor_dict_candidates(item,f"{label}.{key}"))
    return out

def key_variants(key):
    prefixes=["module.","_orig_mod.","model.","net.","ema.","ema_model.","model_ema."]
    variants={key}
    changed=True
    while changed:
        changed=False
        for current in list(variants):
            for prefix in prefixes:
                if current.startswith(prefix):
                    stripped=current[len(prefix):]
                    if stripped not in variants:
                        variants.add(stripped); changed=True
    return variants

def select_state_dict(raw,model):
    model_state=model.state_dict()
    model_keys=set(model_state)
    candidates=tensor_dict_candidates(raw)
    if not candidates and isinstance(raw,dict):
        candidates=[("root",raw)]
    best=None
    for label,candidate in candidates:
        mapped={}; overlap=0; shape_matches=0
        for original,tensor in candidate.items():
            selected=None
            for variant in key_variants(original):
                if variant in model_keys:
                    selected=variant; break
            if selected is None:
                continue
            mapped[selected]=tensor
            overlap += 1
            if tuple(tensor.shape)==tuple(model_state[selected].shape):
                shape_matches += 1
        score=(shape_matches,overlap,-len(candidate))
        if best is None or score>best[0]:
            best=(score,label,mapped)
    if best is None:
        raise RuntimeError("No tensor state dictionary found")
    return best[1],best[2]

def torch_load(path):
    try:
        return torch.load(path,map_location="cpu",weights_only=True)
    except TypeError:
        return torch.load(path,map_location="cpu")

def main():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(MODEL_FILE)
    if not INVENTORY_FILE.is_file():
        raise FileNotFoundError(INVENTORY_FILE)

    inventory=pd.read_csv(INVENTORY_FILE)
    if len(inventory)!=10:
        raise AssertionError(f"Expected 10 inventory rows, found {len(inventory)}")

    module=import_module(MODEL_FILE)
    rows=[]

    for _,item in inventory.iterrows():
        checkpoint=ROOT/str(item["release_file"])
        algorithm=str(item["algorithm"])
        payload=str(item["payload"])
        row={"algorithm":algorithm,"payload":payload,
             "checkpoint":str(checkpoint.relative_to(ROOT)),
             "status":"FAIL","error":""}
        try:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            observed_hash=sha256_file(checkpoint)
            if observed_hash.lower()!=str(item["sha256"]).lower():
                raise RuntimeError("Checkpoint SHA-256 mismatch")

            cfg=module.CFG()
            cfg.algorithm=algorithm
            cfg.payload=payload
            model=module.hsnnet(cfg).cpu()

            total=sum(p.numel() for p in model.parameters())
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
            frontend=sum(p.numel() for p in model.frontend.parameters())

            raw=torch_load(checkpoint)
            state_source,state=select_state_dict(raw,model)
            model_keys=set(model.state_dict())
            state_keys=set(state)
            missing=sorted(model_keys-state_keys)
            unexpected=sorted(state_keys-model_keys)
            shape_mismatches=[
                k for k in model_keys&state_keys
                if tuple(model.state_dict()[k].shape)!=tuple(state[k].shape)
            ]
            if missing or unexpected or shape_mismatches:
                raise RuntimeError(
                    f"state={state_source}; missing={len(missing)}; "
                    f"unexpected={len(unexpected)}; "
                    f"shape_mismatches={len(shape_mismatches)}"
                )

            model.load_state_dict(state,strict=True)
            model.eval()
            with torch.inference_mode():
                output=model(torch.zeros(1,1,256,256,dtype=torch.float32))

            if tuple(output.shape)!=(1,1):
                raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}")
            if total!=EXPECTED_TOTAL or trainable!=EXPECTED_TRAINABLE or frontend!=EXPECTED_FRONTEND:
                raise RuntimeError(
                    f"Parameter count mismatch {total}/{trainable}/{frontend}"
                )

            row.update({
                "checkpoint_sha256":observed_hash,
                "state_source":state_source,
                "total_parameters":total,
                "trainable_parameters":trainable,
                "frontend_parameters":frontend,
                "missing_keys":0,"unexpected_keys":0,"shape_mismatches":0,
                "output_shape":str(tuple(output.shape)),"status":"PASS"
            })
            print(f"PASS | {payload:6s} | {algorithm:11s} | {total:,} parameters")
            del model,raw
        except Exception as exc:
            row["error"]=repr(exc)
            print(f"FAIL | {payload:6s} | {algorithm:11s} | {exc}")
        rows.append(row)

    audit=pd.DataFrame(rows)
    audit.to_csv(
        OUT_DIR/"checkpoint_strict_load_audit.csv",
        index=False,encoding="utf-8-sig"
    )
    passed=audit["status"].eq("PASS").all()
    report="\n".join([
        "HSNNET CHECKPOINT STRICT-LOAD AUDIT","="*76,
        f"Checkpoint conditions : {len(audit)}",
        f"Passed                : {int(audit['status'].eq('PASS').sum())}",
        f"Failed                : {int(audit['status'].eq('FAIL').sum())}",
        f"Total parameters      : {EXPECTED_TOTAL:,}",
        f"Trainable parameters  : {EXPECTED_TRAINABLE:,}",
        f"Frozen front-end      : {EXPECTED_FRONTEND:,}","",
        "FINAL STATUS: "+(
            "PASS — ALL TEN CHECKPOINTS STRICTLY VERIFIED."
            if passed else
            "FAIL — CHECKPOINT AUDIT FAILED."
        )
    ])
    (OUT_DIR/"VERIFY_CHECKPOINTS_REPORT.txt").write_text(report,encoding="utf-8")
    print("\n"+report)
    if not passed:
        raise RuntimeError("Checkpoint strict-load audit failed")

if __name__=="__main__":
    main()
