# filename: scripts/extract_features.py
"""
Main feature‑extraction pipeline for MyMusic.
✅ Uses BasicPitch’s native model path (CoreML/ONNX) automatically.
✅ Compatible with macOS (M3/MPS + CPU fallback).

🧠 NEW LOGIC (OCT 2025)
────────────────────────────
• Removes hardcoded ~/.cache/basic_pitch/basic_pitch.onnx path.
• Relies entirely on utils.validate_model_load() and BasicPitch internal model path.
• Supports any platform transparently.
"""

import argparse
import concurrent.futures
import os
from datetime import datetime
from pathlib import Path

# ✅ Core utilities from utils
from utils.utils import (
    setup_logging,
    is_audio_valid,
    extract_audio_features,
    extract_midi,
    log_performance,
    save_json,
    validate_model_load,
)


def ensure_basicpitch_model_ready():
    """
    Ensures BasicPitch’s platform‑specific model is available.
    On macOS → CoreML (.mlmodel)
    On Linux/Windows → ONNX (.onnx)
    """
    print("[⚙️] Validating BasicPitch model...")
    if not validate_model_load():
        print("[⚠️] Model missing or invalid. Please reinstall BasicPitch: `pip install basic-pitch[onnx]` or `[tf]`.")
    else:
        print("[✅] Model ready for inference.")


def process_file(file_path: str, output_dir: str, log_dir: str):
    """Validate, extract MIDI and audio features for one file."""
    track_name = os.path.basename(file_path)
    start_time = datetime.now()
    try:
        genre = Path(file_path).parent.name
        track_out = Path(output_dir) / genre / Path(file_path).stem
        track_out.mkdir(parents=True, exist_ok=True)

        midi_path = track_out / (Path(file_path).stem + ".mid")
        features_path = track_out / "audio_features.json"
        if midi_path.exists() and features_path.exists():
            status = "✅ already processed"
            raise Exception(status)

        if not is_audio_valid(file_path):
            raise ValueError("Invalid or empty WAV file")

        midi_ok = extract_midi(file_path, str(track_out))
        features = extract_audio_features(file_path)
        if features:
            save_json(features, features_path)

        status = "✅ success" if midi_ok and features else "⚠️ partial"

    except Exception as e:
        status = str(e) if "already processed" in str(e) else f"❌ error: {e}"
    finally:
        end_time = datetime.now()
        log_performance(track_name, start_time, end_time, status, log_dir)


def main():
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    ensure_basicpitch_model_ready()
    log_file = setup_logging(args.log_dir)
    print(f"[🔍] Log file: {log_file}")

    all_wavs = sorted([str(p) for p in Path(args.input_dir).rglob("*.wav")])
    print(f"[🎵] Found {len(all_wavs)} WAV files")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(lambda f: process_file(f, args.output_dir, args.log_dir), all_wavs))


if __name__ == "__main__":
    main()