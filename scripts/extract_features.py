# filename: scripts/extract_features.py
"""
Main feature‑extraction pipeline for MyMusic.
✅ Uses BasicPitch’s native model path (CoreML/ONNX) automatically.
✅ Compatible with macOS (M3/MPS + CPU fallback).
✅ Organizes outputs under per‑genre format folders (wav/json/mid/logs).

🧠 STRUCTURE‑AWARE VERSION – NOV 2025
────────────────────────────
• Preserves genre hierarchy from dataset/
• Groups outputs by format:
    dataset_features/<genre>/{wav,json,mid,logs}/
• Auto‑creates folders safely for each genre.
• Parallelized with ThreadPoolExecutor.
"""

import argparse
import concurrent.futures
import os
import shutil
from datetime import datetime
from pathlib import Path

# ✅ Import from utils
from utils.utils import (
    setup_logging,
    is_audio_valid,
    extract_audio_features,
    extract_midi,
    log_performance,
    save_json,
    validate_model_load,
)


# -----------------------------------------------------------
# 🧩 Ensure BasicPitch model ready
# -----------------------------------------------------------
def ensure_basicpitch_model_ready():
    """Checks that the platform‑specific BasicPitch model exists."""
    print("[⚙️] Validating BasicPitch model...")
    if not validate_model_load():
        print("[⚠️] Model missing or invalid. Please reinstall BasicPitch: `pip install basic-pitch[onnx]` or `[tf]`.")
    else:
        print("[✅] Model ready for inference.")


# -----------------------------------------------------------
# 🎧 Process single file
# -----------------------------------------------------------
def process_file(file_path: str, output_root: str, log_dir: str):
    """
    Process a single .wav file:
    1. Copies input WAV to /wav/
    2. Extracts MIDI → /mid/
    3. Extracts audio features → /json/
    """
    track_name = Path(file_path).stem
    genre = Path(file_path).parents[1].name  # e.g., dataset/classical/wav/file.wav → "classical"
    start_time = datetime.now()

    # Prepare folder structure under output root
    genre_out = Path(output_root) / genre
    wav_dir = genre_out / "wav"
    mid_dir = genre_out / "mid"
    json_dir = genre_out / "json"
    genre_log_dir = genre_out / "logs"

    for d in [wav_dir, mid_dir, json_dir, genre_log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        # 1️⃣ Validate or skip if already processed
        midi_path = mid_dir / f"{track_name}.mid"
        json_path = json_dir / f"{track_name}.json"
        if midi_path.exists() and json_path.exists():
            status = "✅ already processed"
            raise Exception(status)

        if not is_audio_valid(file_path):
            raise ValueError("Invalid or empty WAV file")

        # 2️⃣ Copy source WAV (for traceability)
        dst_wav = wav_dir / f"{track_name}.wav"
        if not dst_wav.exists():
            shutil.copy2(file_path, dst_wav)

        # 3️⃣ Extract MIDI
        midi_ok = extract_midi(file_path, str(mid_dir))

        # 4️⃣ Extract audio features (tempo/onsets)
        features = extract_audio_features(file_path)
        if features:
            save_json(features, json_path)

        status = "✅ success" if midi_ok and features else "⚠️ partial"

    except Exception as e:
        status = str(e) if "already processed" in str(e) else f"❌ error: {e}"
    finally:
        end_time = datetime.now()
        log_performance(track_name, start_time, end_time, status, str(genre_log_dir))


# -----------------------------------------------------------
# 🚀 Main pipeline
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Feature extraction pipeline with per-format outputs")
    parser.add_argument("--input_dir", required=True, help="Input dataset folder (e.g., dataset/)")
    parser.add_argument("--output_dir", required=True, help="Output folder (e.g., dataset_features/)")
    parser.add_argument("--log_dir", required=True, help="Root log directory (e.g., logs/)")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel workers (default=8)")
    args = parser.parse_args()

    # Validate model and prepare logging
    ensure_basicpitch_model_ready()
    log_file = setup_logging(args.log_dir)
    print(f"[🔍] Global log file: {log_file}")

    # Collect all WAVs recursively (only inside wav/ subfolders)
    all_wavs = sorted([str(p) for p in Path(args.input_dir).rglob("*.wav")])
    print(f"[🎵] Found {len(all_wavs)} WAV files under {args.input_dir}")

    if not all_wavs:
        print("[⚠️] No WAV files found.")
        return

    # Parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(lambda f: process_file(f, args.output_dir, args.log_dir), all_wavs))

    print("✅ Feature extraction completed.")


# -----------------------------------------------------------
# 🏁 Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()