# filename: scripts/fix_output_structure.py

import os
import json
import shutil
from pathlib import Path

INPUT_ROOT = Path("dataset_features")
REQUIRED_JSON_KEYS = {"tempo", "onsets"}

def is_valid_midi(file_path: Path) -> bool:
    return file_path.exists() and file_path.stat().st_size > 0

def is_valid_json(file_path: Path) -> bool:
    try:
        with open(file_path) as f:
            data = json.load(f)
        return REQUIRED_JSON_KEYS.issubset(data)
    except Exception:
        return False

def move_and_validate():
    print(f"🔧 Fixing structure under: {INPUT_ROOT}")
    genres = [p for p in INPUT_ROOT.iterdir() if p.is_dir()]

    for genre_dir in genres:
        print(f"🎵 Genre: {genre_dir.name}")
        mid_dir = genre_dir / "mid"
        json_dir = genre_dir / "json"
        mid_dir.mkdir(exist_ok=True)
        json_dir.mkdir(exist_ok=True)

        for subdir in genre_dir.iterdir():
            if not subdir.is_dir() or subdir.name in {"mid", "json"}:
                continue

            stem = subdir.name
            mid_file = subdir / f"{stem}_basic_pitch.mid"
            json_file = subdir / "audio_features.json"

            if mid_file.exists():
                dest = mid_dir / f"{stem}.mid"
                shutil.move(str(mid_file), str(dest))
                valid = is_valid_midi(dest)
                print(f"  🎼 {stem}.mid → {'✅ valid' if valid else '❌ invalid'}")

            if json_file.exists():
                dest = json_dir / f"{stem}.json"
                shutil.move(str(json_file), str(dest))
                valid = is_valid_json(dest)
                print(f"  📊 {stem}.json → {'✅ valid' if valid else '❌ invalid'}")

            # Optionally remove the original subdir if empty
            try:
                subdir.rmdir()
            except OSError:
                pass

if __name__ == "__main__":
    move_and_validate()