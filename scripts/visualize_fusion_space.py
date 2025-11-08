# =============================================
# File: scripts/visualize_fusion_space.py
# Stage: 4 (Fusion – Visualization)
# Description:
#   Generates 2D scatterplots (t-SNE) and a CSV file of
#   latent embeddings per genre for quality inspection.
#
# Inputs:
#   - fusion_models/cache/embeddings_pca.npy
#   - fusion_models/cache/embeddings_tsne.npy
#   - fusion_models/cache/fusion_dataset.jsonl
#
# Outputs:
#   - fusion_models/cache/embeddings_2d.csv
#   - fusion_models/cache/plots/tsne_2d.png
#   - logs/fusion_vis_*.jsonl
# =============================================

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for macOS server/CLI runs
import matplotlib.pyplot as plt

# --- Project roots ---
ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "fusion_models"
CACHE_DIR = FUSION_DIR / "cache"
PLOTS_DIR = CACHE_DIR / "plots"
LOGS_DIR = ROOT / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Input/output paths ---
EMB_PCA_NPY  = CACHE_DIR / "embeddings_pca.npy"
EMB_TSNE_NPY = CACHE_DIR / "embeddings_tsne.npy"
DATASET_JSONL = CACHE_DIR / "fusion_dataset.jsonl"
OUT_CSV = CACHE_DIR / "embeddings_2d.csv"
RUN_LOG = LOGS_DIR / f"fusion_vis_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

# --- Logger / JSONL fallback ---
try:
    sys.path.append(str(ROOT))
    from scripts.utils.utils import get_logger, jsonl_open
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
        def __init__(self, path): self.path = Path(path)
        def __enter__(self): self.f = open(self.path, "a", encoding="utf-8"); return self
        def write(self, obj: Dict[str, Any]): self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        def __exit__(self, *a): self.f.close()

logger = get_logger("visualize_fusion_space")


# -------------------------
# Helper functions
# -------------------------
def load_dataset(jsonl_path: Path):
    """Return track IDs and genres from fusion_dataset.jsonl."""
    ids, genres = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["track_id"])
            genres.append(row["genre"])
    return ids, genres


# -------------------------
# Main visualization logic
# -------------------------
def main():
    if not (EMB_PCA_NPY.exists() and EMB_TSNE_NPY.exists() and DATASET_JSONL.exists()):
        logger.error("❌ Missing embeddings or dataset. Run previous Stage 4 steps first.")
        sys.exit(2)

    ids, genres = load_dataset(DATASET_JSONL)
    X_pca  = np.load(EMB_PCA_NPY)
    X_tsne = np.load(EMB_TSNE_NPY)

    # Create dataframe for easy export
    df = pd.DataFrame({
        "track_id": ids,
        "genre": genres,
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1]
    })
    df.to_csv(OUT_CSV, index=False)

    # --- Plot t-SNE scatter by genre ---
    plt.figure(figsize=(8, 6))
    unique_genres = sorted(set(genres))
    for g in unique_genres:
        mask = [i for i, gg in enumerate(genres) if gg == g]
        plt.scatter(df.loc[mask, "x"], df.loc[mask, "y"], label=g, s=18, alpha=0.75)
    plt.title("t-SNE – Fusion Genre Space (2D)")
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    plt.legend()
    plt.tight_layout()

    fig_path = PLOTS_DIR / "tsne_2d.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    with jsonl_open(RUN_LOG) as jl:
        jl.write({
            "event": "vis_done",
            "csv": str(OUT_CSV),
            "tsne_png": str(fig_path),
            "n_points": int(len(df)),
            "genres": sorted(set(genres))
        })
    logger.info(f"✅ Visualization complete. Saved {len(df)} points → {fig_path}")

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    main()