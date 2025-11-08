# ============================================
# File: scripts/rename_mid_files.py
# Purpose: Align MIDI filenames with JSON names from Stage 2
# Run from: MyMusic/
# ============================================

import os
import json
import time
import difflib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET_FEATURES = ROOT / "dataset_features"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"rename_mid_files_{TIMESTAMP}.jsonl"

# ---------- Helpers ----------
def normalize(name: str) -> str:
    """Normalize a filename for fuzzy comparison."""
    name = name.lower().replace("_basic_pitch", "")
    chars_to_strip = [",", "’", "'", "‘", "“", "”", "(", ")", "-", "–", "—", ".", "_"]
    for c in chars_to_strip:
        name = name.replace(c, " ")
    return " ".join(name.split())

def log_action(mid_file, new_name, action, similarity):
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mid_file": str(mid_file),
        "new_name": new_name,
        "action": action,
        "similarity": round(similarity, 3),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ---------- Core ----------
def rename_mid_files(genre_dir: Path):
    genre = genre_dir.name
    mid_dir = genre_dir / "mid"
    json_dir = genre_dir / "json"

    if not mid_dir.exists() or not json_dir.exists():
        print(f"⚠️  Skipping {genre} (missing subfolders)")
        return

    mids = list(mid_dir.glob("*.mid"))
    jsons = list(json_dir.glob("*.json"))

    if not mids or not jsons:
        print(f"⚠️  No files for {genre}")
        return

    json_names = {normalize(j.stem): j for j in jsons}

    print(f"\n🎼 Processing genre: {genre} ({len(mids)} MIDI files)")
    for mid_file in tqdm(mids):
        norm_mid = normalize(mid_file.stem)
        # Find best match among JSON stems
        best_match = difflib.get_close_matches(norm_mid, json_names.keys(), n=1, cutoff=0.8)
        if not best_match:
            log_action(mid_file, None, "no_match", 0.0)
            continue

        best_key = best_match[0]
        similarity = difflib.SequenceMatcher(None, norm_mid, best_key).ratio()
        json_path = json_names[best_key]
        json_stem = json_path.stem

        if mid_file.stem == json_stem:
            log_action(mid_file, json_stem, "already_matched", 1.0)
            continue

        new_name = mid_dir / f"{json_stem}.mid"
        try:
            mid_file.rename(new_name)
            log_action(mid_file, new_name.name, "renamed", similarity)
        except Exception as e:
            log_action(mid_file, str(e), "error", similarity)

# ---------- Runner ----------
def main():
    print("🎯 Stage 3 Prep: Aligning MIDI and JSON filenames\n")
    start = time.time()

    genres = [g for g in DATASET_FEATURES.iterdir() if g.is_dir()]
    for g in genres:
        rename_mid_files(g)

    elapsed = round(time.time() - start, 2)
    print(f"\n✅ Completed renaming in {elapsed}s.")
    print(f"🧾 Log → {LOG_FILE}")


if __name__ == "__main__":
    main()