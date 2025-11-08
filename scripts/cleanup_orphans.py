# filename: scripts/cleanup_orphans.py
import shutil
from pathlib import Path

ROOT = Path("dataset_features")

def is_orphan_folder(folder: Path) -> bool:
    """Returns True if folder is NOT one of the known genre folders."""
    genre_names = {"jazz", "trance", "salsa", "hijaz", "classical"}
    return folder.is_dir() and folder.name not in genre_names

def main():
    orphans = [p for p in ROOT.iterdir() if is_orphan_folder(p)]
    for folder in orphans:
        print(f"🧹 Deleting orphan folder: {folder}")
        shutil.rmtree(folder)

if __name__ == "__main__":
    main()