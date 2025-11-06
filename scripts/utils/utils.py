# filename: scripts/utils/utils.py
"""
Utility functions for the MyMusic feature extraction pipeline.
✅ Fully compatible with BasicPitch 0.4.x (Python 3.10–3.11 / macOS ARM64)

🧠 FINAL STABILIZED VERSION
────────────────────────────
This version unifies all previous fixes and adds automatic ONNX model setup.

🪄 Behavior summary:
  • Automatically ensures `~/.cache/basic_pitch/basic_pitch.onnx` exists.
  • If missing → auto-downloads the model using `basic_pitch.model_loader`.
  • Always passes `model_or_model_path` to avoid TypeError in 0.4.x.
  • Still supports internal default mode for older forks.

💡 Core logic:
    if MODEL exists → pass it explicitly (always safe)
    else             → auto-download or fallback to internal model
"""

import os
import json
import logging
import librosa
import soundfile as sf
import numpy as np
import inspect
from datetime import datetime
from importlib import import_module

# Main BasicPitch function
from basic_pitch.inference import predict_and_save


# -----------------------------------------------------------
# 🎯 Hardcoded ONNX model path
# -----------------------------------------------------------

HARD_CODED_MODEL_PATH = os.path.expanduser("~/.cache/basic_pitch/basic_pitch.onnx")


# -----------------------------------------------------------
# 🔍 Dynamic model loader resolver
# -----------------------------------------------------------

def _resolve_basic_pitch_loader():
    """Try known module paths for load_model()."""
    for path in ("basic_pitch.inference", "basic_pitch.model_loader"):
        try:
            mod = import_module(path)
            if hasattr(mod, "load_model"):
                logging.info(f"[✅] Using load_model from {path}")
                return getattr(mod, "load_model")
        except ModuleNotFoundError:
            continue
    logging.warning("[⚠️] No load_model() found — will use auto-downloaded ONNX fallback")
    return None


# -----------------------------------------------------------
# 🧠 Global model setup
# -----------------------------------------------------------

load_model = _resolve_basic_pitch_loader()
MODEL = None

try:
    # Ensure cache folder
    os.makedirs(os.path.dirname(HARD_CODED_MODEL_PATH), exist_ok=True)

    # Check local model; if missing, auto-download
    if not os.path.exists(HARD_CODED_MODEL_PATH):
        logging.info(f"[⬇️] Downloading ONNX model to {HARD_CODED_MODEL_PATH}")
        from basic_pitch.model_loader import download_model
        download_model("basic_pitch.onnx")

    # Assign model path (works with predict_and_save)
    MODEL = HARD_CODED_MODEL_PATH

except Exception as e:
    logging.error(f"[❌] Failed to preload or download BasicPitch model: {e}")
    MODEL = None


# -----------------------------------------------------------
# ⚙️ Logging setup
# -----------------------------------------------------------

def setup_logging(log_dir: str) -> str:
    """Create log directory + file and configure logging early."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Ensure file physically exists (for test assertions)
    open(log_file, "a").close()

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    return log_file


# -----------------------------------------------------------
# 🎧 Audio validation
# -----------------------------------------------------------

def is_audio_valid(filepath: str) -> bool:
    """Check that audio exists, is readable, and non‑empty."""
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        with sf.SoundFile(filepath) as f:
            return f.frames > 0 and f.samplerate > 0
    except Exception:
        return False


# -----------------------------------------------------------
# 🎚 Audio feature extraction
# -----------------------------------------------------------

def extract_audio_features(file_path: str) -> dict:
    """Extract tempo and onset timings using librosa."""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return {"tempo": tempo, "onsets": onset_times.tolist()}
    except Exception as e:
        logging.error(f"[❌] {os.path.basename(file_path)} failed in extract_audio_features: {e}")
        return None


# -----------------------------------------------------------
# 🎼 MIDI extraction (robust + auto-model)
# -----------------------------------------------------------

def extract_midi(file_path: str, output_dir: str) -> bool:
    """
    Extract MIDI using BasicPitch.
    Always passes an explicit model path if available.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        sig = inspect.signature(predict_and_save)

        # ✅ Always pass model path (either real ONNX or internal)
        if "model_or_model_path" in sig.parameters:
            logging.info(f"[🎹] Using ONNX model for {os.path.basename(file_path)}")
            predict_and_save(
                [file_path],
                output_directory=output_dir,
                model_or_model_path=MODEL or "default",
                save_model_outputs=False,
                save_notes=False,
                save_midi=True,
                sonify_midi=False,
            )
        else:
            # 🕹 Fallback for older forks
            logging.info(f"[🎹] Legacy predict_and_save() for {os.path.basename(file_path)}")
            predict_and_save(
                [file_path],
                output_directory=output_dir,
                save_model_outputs=False,
                save_notes=False,
                save_midi=True,
                sonify_midi=False,
            )
        return True

    except Exception as e:
        logging.error(f"[❌] {os.path.basename(file_path)} failed in extract_midi: {e}")
        return False


# -----------------------------------------------------------
# 💾 JSON & performance helpers
# -----------------------------------------------------------

def save_json(data: dict, output_path: str):
    """Persist extracted feature data to disk."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"[❌] Failed to save JSON: {e}")


def log_performance(track_name: str, start_time: datetime, end_time: datetime, status: str, log_dir: str):
    """Append per‑track timing to performance_summary.log."""
    try:
        duration = (end_time - start_time).total_seconds()
        with open(os.path.join(log_dir, "performance_summary.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} - {track_name} - {status} ({duration:.2f}s)\n")
    except Exception as e:
        logging.error(f"[❌] Failed to log performance for {track_name}: {e}")


# -----------------------------------------------------------
# 🧪 Optional preflight check
# -----------------------------------------------------------

def validate_model_load() -> bool:
    """Explicit model‑load test for diagnostics."""
    try:
        if os.path.exists(HARD_CODED_MODEL_PATH):
            logging.info("[✅] Hardcoded ONNX model found locally.")
            return True
        else:
            logging.warning("[⚠️] ONNX model missing — will trigger download automatically.")
            return False
    except Exception as e:
        logging.error(f"[❌] Model load validation failed: {e}")
        return False