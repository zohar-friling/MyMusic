# filename: scripts/extract_features.py
"""
Main feature‑extraction pipeline for MyMusic.
✅ Now includes automatic BasicPitch ONNX model verification + download.
✅ Compatible with basic_pitch 0.4.x on macOS (M3/MPS + CPU fallback).

🧠 NEW FEATURES
────────────────────────────
1️⃣ Automatically checks for the ONNX model in ~/.cache/basic_pitch/basic_pitch.onnx  
2️⃣ Downloads it automatically if missing (no manual setup needed)
3️⃣ Works seamlessly with both test_extract_features.py and production datasets

💡 The logic is imported from utils.utils, which handles dynamic BasicPitch compatibility.
"""

import argparse
import concurrent.futures
import os
from datetime import datetime
from pathlib import Path

# ✅ Utility functions (now include model auto‑download logic)
from utils.utils import (
    setup_logging,
    is_audio_valid,
    extract_audio_features,
    extract_midi,
    log_performance,
    save_json,
    validate_model_load,   # ✅ new addition to auto‑validate model
)

# -----------------------------------------------------------
# ⚙️ Automatic model setup
# -----------------------------------------------------------

def ensure_basicpitch_model_ready():
    """
    Ensures that the ONNX model for BasicPitch exists locally.
    If not, downloads it automatically into ~/.cache/basic_pitch/.
    """
    from basic_pitch.model_loader import download_model

    model_path = os.path.expanduser("~/.cache/basic_pitch/basic_pitch.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[⬇️] ONNX model not found — downloading to {model_path}")
        try:
            download_model("basic_pitch.onnx")
            print("[✅] Model downloaded successfully.")
        except Exception as e:
            print(f"[❌] Model download failed: {e}")
    else:
        print(f"[✅] ONNX model already present at {model_path}")

    # Double‑check using validate_model_load() from utils
    if not validate_model_load():
        print("[⚠️] Warning: model load validation failed. BasicPitch may fallback internally.")


# -----------------------------------------------------------
# 🧱 File‑level processing (per track)
# -----------------------------------------------------------

def process_file(file_path: str, output_dir: str, log_dir: str):
    """Process a single audio file: validate, extract MIDI and audio features."""

    track_name = os.path.basename(file_path)
    start_time = datetime.now()

    try:
        genre = Path(file_path).parent.name
        track_output_dir = Path(output_dir) / genre / Path(file_path).stem
        track_output_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Idempotent skip if both outputs exist
        midi_path = track_output_dir / (Path(file_path).stem + ".mid")
        features_path = track_output_dir / "audio_features.json"
        if midi_path.exists() and features_path.exists():
            status = "✅ already processed"
            raise Exception(status)

        # 🧪 Validate audio integrity
        if not is_audio_valid(file_path):
            raise ValueError("Invalid or empty WAV file")

        # 🎼 Extract MIDI (BasicPitch + ONNX model)
        midi_success = extract_midi(file_path, str(track_output_dir))

        # 🧠 Extract audio features (tempo + onsets)
        features = extract_audio_features(file_path)
        if features:
            save_json(features, features_path)

        status = "✅ success" if midi_success and features else "⚠️ partial"

    except Exception as e:
        status = str(e) if "already processed" in str(e) else f"❌ error: {e}"

    finally:
        # ⏱ Always log runtime + status
        end_time = datetime.now()
        log_performance(track_name, start_time, end_time, status, log_dir)


# -----------------------------------------------------------
# 🚀 Main orchestration entrypoint
# -----------------------------------------------------------

def main():
    """
    Orchestrates:
    - Model verification / download
    - Argument parsing
    - WAV file discovery
    - Parallel feature extraction
    """
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--input_dir", required=True, help="Path to input dataset")
    parser.add_argument("--output_dir", required=True, help="Path to output dataset features")
    parser.add_argument("--log_dir", required=True, help="Path to log directory")
    parser.add_argument("--max_workers", type=int, default=8, help="Max concurrent workers")
    args = parser.parse_args()

    # ✅ Ensure model exists before doing any processing
    ensure_basicpitch_model_ready()

    # 🧾 Logging setup
    log_file = setup_logging(args.log_dir)
    print(f"[🔍] Log file: {log_file}")

    # 🎧 Collect all WAV files recursively
    input_dir = Path(args.input_dir)
    all_wavs = sorted([str(p) for p in input_dir.rglob("*.wav")])
    print(f"[🎵] Found {len(all_wavs)} WAV files")

    # 🧵 Parallel multi‑threaded processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(executor.map(lambda f: process_file(f, args.output_dir, args.log_dir), all_wavs))


if __name__ == "__main__":
    main()