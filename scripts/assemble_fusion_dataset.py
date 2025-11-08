# =============================================
# File: scripts/assemble_fusion_dataset.py
# Stage: 4 (Fusion – Data Assembly & Normalization)
# Description: Aggregates *.analysis.json from fusion_models/<genre>/
#              into a unified, normalized dataset for embedding/training.
#              Parallel I/O, tqdm progress, JSONL logging, idempotent outputs.
# =============================================

import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# --- Project roots ---
ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "fusion_models"
CACHE_DIR = FUSION_DIR / "cache"
LOGS_DIR = ROOT / "logs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Optional utils fallback ---
try:
    sys.path.append(str(ROOT))
    from scripts.utils.utils import (
        get_logger,
        jsonl_open,
        write_json,
        read_json,
        list_genre_dirs,
        standardize_feature_vector,
        collect_feature_schema,
    )
except Exception:
    def get_logger(name: str):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(ch)
        return logger

    class jsonl_open:
        def __init__(self, path):
            self.path = Path(path)
            self.f = None
        def __enter__(self):
            self.f = open(self.path, 'a', encoding='utf-8')
            return self
        def write(self, obj: Dict[str, Any]):
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        def __exit__(self, exc_type, exc, tb):
            if self.f:
                self.f.close()

    def write_json(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def read_json(path: Path) -> Any:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_genre_dirs(base: Path) -> List[Path]:
        return [p for p in base.iterdir() if p.is_dir() and p.name not in {"cache"}]

    def collect_feature_schema(records: List[Dict[str, Any]]) -> List[str]:
        keys = set()
        for r in records:
            feats = r.get('features', {})
            for k, v in feats.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    keys.add(k)
        return sorted(keys)

    def standardize_feature_vector(vec: Dict[str, Any], schema: List[str], stats: Dict[str, Tuple[float, float]]):
        out = {}
        for k in schema:
            x = vec.get(k, None)
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                out[k] = 0.0
                continue
            mu, sigma = stats.get(k, (0.0, 1.0))
            sigma = sigma if sigma not in (0.0, None) else 1.0
            out[k] = float((x - mu) / sigma)
        return out

logger = get_logger("assemble_fusion_dataset")
DATASET_JSONL = CACHE_DIR / "fusion_dataset.jsonl"
SCHEMA_JSON = CACHE_DIR / "fusion_schema.json"
STATS_JSON = CACHE_DIR / "fusion_stats.json"
RUN_LOG = LOGS_DIR / f"fusion_assemble_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

GENRES = [d.name for d in list_genre_dirs(FUSION_DIR)]

def _find_analysis_files() -> List[Tuple[str, Path]]:
    files = []
    for g in GENRES:
        gdir = FUSION_DIR / g
        for p in gdir.glob("*.analysis.json"):
            files.append((g, p))
    return files

def _load_one(genre: str, path: Path) -> Dict[str, Any]:
    try:
        rec = read_json(path)
        feats = rec.get("features", {})
        if not feats:
            feats = {k: v for k, v in rec.items() if isinstance(v, (int, float))}
        track_id = rec.get("track_id") or path.stem.replace(".analysis", "")
        return {
            "track_id": track_id,
            "genre": genre,
            "features": feats,
            "source_path": str(path)
        }
    except Exception as e:
        return {"error": str(e), "genre": genre, "source_path": str(path)}

def main(max_workers: int = 16):
    logger.info("Scanning analysis files…")
    pairs = _find_analysis_files()
    if not pairs:
        logger.error("No *.analysis.json files found under fusion_models/<genre>/")
        sys.exit(2)

    records = []
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex, jsonl_open(RUN_LOG) as jl:
        futs = [ex.submit(_load_one, g, p) for g, p in pairs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Load analyses"):
            rec = fut.result()
            if 'error' in rec:
                errors += 1
                jl.write({"event": "load_error", "rec": rec})
                continue
            records.append(rec)

    if not records:
        logger.error("All loads failed; aborting.")
        sys.exit(3)

    logger.info("Collecting feature schema…")
    schema = collect_feature_schema(records)
    write_json(SCHEMA_JSON, {"feature_schema": schema})

    logger.info("Computing feature statistics…")
    stats = {}
    for k in schema:
        vals = []
        for r in records:
            v = r['features'].get(k, None)
            if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                vals.append(float(v))
        mu = float(np.mean(vals)) if vals else 0.0
        sd = float(np.std(vals)) if vals else 1.0
        stats[k] = [mu, sd]
    write_json(STATS_JSON, {k: {"mean": v[0], "std": v[1]} for k, v in stats.items()})

    logger.info("Normalizing and writing fusion_dataset.jsonl…")
    with open(DATASET_JSONL, 'w', encoding='utf-8') as f:
        for r in tqdm(records, desc="Normalize+Write"):
            norm = standardize_feature_vector(r['features'], schema, {k: (v[0], v[1]) for k, v in stats.items()})
            out = {"track_id": r['track_id'], "genre": r['genre'], "x": norm}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with jsonl_open(RUN_LOG) as jl:
        jl.write({
            "event": "assemble_done",
            "files": len(pairs),
            "ok": len(records),
            "errors": errors,
            "schema_size": len(schema),
            "dataset_jsonl": str(DATASET_JSONL)
        })
    logger.info("Assembly complete.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    main(max_workers=args.workers)