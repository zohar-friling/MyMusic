# =============================================
# File: scripts/train_fusion_model.py
# Stage: 4 (Fusion – Transformation Network)
# Description:
#   Trains a conditional translator that maps latent embeddings
#   from a source genre to a target genre embedding. It reuses the
#   AutoEncoder encoder from fusion_core.pt and appends the trained
#   translator state_dict + metadata back into the same checkpoint.
#
# Inputs:
#   - fusion_models/cache/fusion_dataset.jsonl   (from assemble_fusion_dataset.py)
#   - fusion_models/fusion_core.pt               (from build_genre_embeddings.py)
#
# Outputs:
#   - fusion_models/fusion_core.pt  (augmented with translator)
#   - logs/fusion_train_*.jsonl
# =============================================

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- Project roots ---
ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = ROOT / "fusion_models"
CACHE_DIR = FUSION_DIR / "cache"
LOGS_DIR = ROOT / "logs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Utils fallback (logger + jsonl) ---
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

logger = get_logger("train_fusion_model")

# --- Files ---
AE_MODEL_PT = FUSION_DIR / "fusion_core.pt"                 # produced in Step 2
DATASET_JSONL = CACHE_DIR / "fusion_dataset.jsonl"          # produced in Step 1
RUN_LOG = LOGS_DIR / f"fusion_train_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

# --- Genre config (order fixed to keep IDs stable) ---
GENRE_ORDER = ["classical", "jazz", "hijaz", "salsa", "trance"]
GENRE_TO_ID = {g: i for i, g in enumerate(GENRE_ORDER)}
N_GENRES = len(GENRE_ORDER)


# -------------------------
#  Utilities
# -------------------------
def one_hot(idx: int, n: int) -> torch.Tensor:
    v = torch.zeros(n, dtype=torch.float32)
    v[idx] = 1.0
    return v


def load_dataset(jsonl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return standardized feature matrix X (already normalized numbers in jsonl) and genre ids y."""
    X, y = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            x = row["x"]
            X.append([x[k] for k in sorted(x.keys())])
            y.append(GENRE_TO_ID.get(row["genre"], -1))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def load_autoencoder(path: Path):
    """Load AE checkpoint dictionary saved by build_genre_embeddings.py"""
    ckpt = torch.load(path, map_location="cpu")
    required = ["state_dict", "input_dim", "latent_dim", "scaler_mean", "scaler_scale"]
    for k in required:
        if k not in ckpt:
            raise RuntimeError(f"fusion_core.pt missing key: {k}")
    return ckpt


# -------------------------
#  Models
# -------------------------
class AEEncoder(nn.Module):
    """Mirror the encoder architecture used in build_genre_embeddings.py"""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, x):
        return self.net(x)


class Translator(nn.Module):
    """Conditional translator: z, src_onehot, tgt_onehot -> z_hat (latent in target style)"""
    def __init__(self, latent_dim: int, n_genres: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + n_genres * 2, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, z, src_onehot, tgt_onehot):
        x = torch.cat([z, src_onehot, tgt_onehot], dim=-1)
        return self.fc(x)


# -------------------------
#  Training
# -------------------------
def train(latent_dim: int = 256,
          epochs: int = 30,
          lr: float = 1e-3,
          batch: int = 256,
          grad_clip: float = 1.0):

    if not AE_MODEL_PT.exists():
        logger.error("Missing fusion_core.pt from build_genre_embeddings.py")
        sys.exit(2)
    if not DATASET_JSONL.exists():
        logger.error("Missing fusion_dataset.jsonl. Run assemble_fusion_dataset.py first.")
        sys.exit(3)

    # Load AE & rebuild encoder
    ae = load_autoencoder(AE_MODEL_PT)
    input_dim = int(ae["input_dim"])
    ckpt_state = ae["state_dict"]

    encoder = AEEncoder(input_dim=input_dim, latent_dim=latent_dim)
    enc_sd = {k.replace("encoder.", ""): v for k, v in ckpt_state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder.to(device).eval()

    # Load dataset (already normalized per assemble_fusion_dataset stats)
    X, y = load_dataset(DATASET_JSONL)
    scaler_mean = np.array(ae["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(ae["scaler_scale"], dtype=np.float32)
    X = (X - scaler_mean) / np.maximum(scaler_scale, 1e-8)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.int64, device=device)

    # Precompute latent embeddings Z
    with torch.no_grad():
        Z = encoder(X)

    # Translator + optimizer
    translator = Translator(latent_dim=latent_dim, n_genres=N_GENRES).to(device)
    opt = torch.optim.Adam(translator.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Precompute per-genre centroids in latent space (updated each epoch)
    def compute_centroids(Z_latent: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
        cents = []
        for gid in range(N_GENRES):
            mask = (y_labels == gid)
            if torch.any(mask):
                cents.append(Z_latent[mask].mean(dim=0))
            else:
                cents.append(torch.zeros_like(Z_latent[0]))
        return torch.stack(cents, dim=0)  # [N_GENRES, latent_dim]

    n = Z.size(0)
    logger.info(f"Training translator on {n} embeddings…")
    with jsonl_open(RUN_LOG) as jl:
        for epoch in range(1, epochs + 1):
            # Recompute centroids each epoch for stability
            centroids = compute_centroids(Z, y)

            perm = torch.randperm(n, device=device)
            total = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                z_src = Z[idx]                      # [B, latent]
                src = y[idx]                        # [B]

                # Sample targets != source
                tgt = torch.randint(low=0, high=N_GENRES, size=src.shape, device=device)
                tgt = torch.where(tgt == src, (tgt + 1) % N_GENRES, tgt)

                # One-hot encode
                src_oh = torch.stack([one_hot(int(s), N_GENRES) for s in src.tolist()]).to(device)
                tgt_oh = torch.stack([one_hot(int(t), N_GENRES) for t in tgt.tolist()]).to(device)

                # Predict target latent
                z_hat = translator(z_src, src_oh, tgt_oh)           # [B, latent]
                z_tgt_centroid = centroids[tgt]                     # [B, latent]

                # Loss: bring predicted latent close to target centroid
                loss = loss_fn(z_hat, z_tgt_centroid)

                # Optimize
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), grad_clip)
                opt.step()

                total += float(loss.detach().cpu()) * z_src.size(0)

            epoch_loss = total / n
            jl.write({"event": "translator_epoch", "epoch": epoch, "loss": epoch_loss})
            logger.info(f"Epoch {epoch}/{epochs} – translator loss: {epoch_loss:.6f}")

    # Augment and resave fusion_core.pt with translator
    ae_aug = torch.load(AE_MODEL_PT, map_location="cpu")
    ae_aug["translator_state_dict"] = translator.state_dict()
    ae_aug["genre_order"] = GENRE_ORDER
    torch.save(ae_aug, AE_MODEL_PT)
    logger.info("Translator trained and appended to fusion_core.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()
    train(latent_dim=args.latent,
          epochs=args.epochs,
          lr=args.lr,
          batch=args.batch,
          grad_clip=args.grad_clip)