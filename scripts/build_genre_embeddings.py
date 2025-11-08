# =============================================
# File: scripts/build_genre_embeddings.py
# Stage: 4 (Fusion – Embedding Construction & Visualization)
# Description: Loads fusion_dataset.jsonl, scales features,
#              builds PCA/t-SNE, trains AutoEncoder (fusion_core.pt)
# =============================================

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn

# --- Directories ---
ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "fusion_models"
CACHE_DIR = FUSION_DIR / "cache"
LOGS_DIR = ROOT / "logs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Utils Fallback ---
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
        def __init__(self, path):
            self.path = Path(path)
        def __enter__(self):
            self.f = open(self.path, "a", encoding="utf-8");  return self
        def write(self, obj: Dict[str, Any]):  self.f.write(json.dumps(obj, ensure_ascii=False)+"\n")
        def __exit__(self, *a):  self.f.close()

logger = get_logger("build_genre_embeddings")

# --- Paths ---
DATASET_JSONL = CACHE_DIR / "fusion_dataset.jsonl"
EMB_PCA_NPY   = CACHE_DIR / "embeddings_pca.npy"
EMB_TSNE_NPY  = CACHE_DIR / "embeddings_tsne.npy"
AE_MODEL_PT   = FUSION_DIR / "fusion_core.pt"
AE_SUMMARY_JSON = FUSION_DIR / f"fusion_model_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
RUN_LOG = LOGS_DIR / f"fusion_embeddings_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

# --- Data Loader ---
def load_dataset(path: Path):
    X, y, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            x = row["x"]
            X.append([x[k] for k in sorted(x.keys())])
            y.append(row["genre"])
            ids.append(row["track_id"])
    features = sorted(list(x.keys()))
    return np.array(X, np.float32), y, ids, features

# --- AutoEncoder ---
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def seed_all(seed=1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- Main ---
def main(latent_dim=256, epochs=40, lr=1e-3, batch_size=256):
    seed_all()

    if not DATASET_JSONL.exists():
        logger.error("Missing fusion_dataset.jsonl — run assemble_fusion_dataset.py first.")
        sys.exit(2)

    X, y, ids, features = load_dataset(DATASET_JSONL)
    logger.info(f"Loaded dataset {X.shape}, features={len(features)}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    logger.info("Running PCA…")
    pca = PCA(n_components=min(64, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    np.save(EMB_PCA_NPY, X_pca)

    # t-SNE
    logger.info("Running t-SNE (2D)…")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", n_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(X_pca)
    np.save(EMB_TSNE_NPY, X_tsne)

    # AutoEncoder
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoEncoder(X_scaled.shape[1], latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    n = len(X_tensor)

    logger.info(f"Training AutoEncoder on {n} samples…")
    with jsonl_open(RUN_LOG) as jl:
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n)
            total = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch = X_tensor[idx].to(device)
                opt.zero_grad()
                recon, _ = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            epoch_loss = total / n
            jl.write({"event": "ae_epoch", "epoch": epoch, "loss": epoch_loss})
            logger.info(f"Epoch {epoch}/{epochs} – loss {epoch_loss:.6f}")

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X_scaled.shape[1],
        "latent_dim": latent_dim,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_order": features,
        "pca_var_ratio": getattr(pca, "explained_variance_ratio_", []).tolist()
    }, AE_MODEL_PT)

    with open(AE_SUMMARY_JSON, "w") as f:
        json.dump({
            "input_dim": int(X_scaled.shape[1]),
            "latent_dim": latent_dim,
            "epochs": epochs,
            "device": str(device),
            "dataset_size": int(n)
        }, f, indent=2)

    logger.info("Embeddings + AutoEncoder saved successfully.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    main(latent_dim=args.latent, epochs=args.epochs, lr=args.lr, batch_size=args.batch)