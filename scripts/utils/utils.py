# filename: scripts/utils/utils.py
"""
Utility functions for the MyMusic feature extraction pipeline.
✅ Fully compatible with BasicPitch 0.4.x (Python 3.10–3.11 / macOS ARM64, CoreML backend)

🧠 UPDATED VERSION – OCT 2025
────────────────────────────
This update replaces the deprecated ONNX hardcoded path logic with an official
cross‑platform model reference (`ICASSP_2022_MODEL_PATH`) provided by the BasicPitch library.

🪄 Behavior summary:
  • Automatically uses the correct model file for each OS:
      macOS  →  CoreML (.mlmodel)
      Linux  →  ONNX (.onnx)
      Windows →  ONNX (.onnx)
  • No manual download or cache validation required.
  • Works natively with `predict_and_save()` and passes the right model path.

💡 Core logic:
    from basic_pitch.models import ICASSP_2022_MODEL_PATH
    MODEL = ICASSP_2022_MODEL_PATH
"""

import os
import json
import logging
import librosa
import soundfile as sf
import numpy as np
import inspect
from datetime import datetime

# ✅ Import the main inference function and platform‑specific model path
from basic_pitch.inference import predict_and_save
from basic_pitch.models import ICASSP_2022_MODEL_PATH

# ✅ Always rely on the OS‑specific model path provided by the library
MODEL = ICASSP_2022_MODEL_PATH


# -----------------------------------------------------------
# ⚙️ Logging setup
# -----------------------------------------------------------
def setup_logging(log_dir: str) -> str:
    """Create log directory + file and configure logging early."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
# 🎚 Audio feature extraction (Librosa)
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
# 🎼 MIDI extraction (robust + CoreML/ONNX auto‑selection)
# -----------------------------------------------------------
def extract_midi(file_path: str, output_dir: str) -> bool:
    """
    Extract MIDI using BasicPitch.
    Automatically uses the correct model for the current platform.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        sig = inspect.signature(predict_and_save)
        if "model_or_model_path" in sig.parameters:
            logging.info(f"[🎹] Using platform‑native model for {os.path.basename(file_path)}")
            predict_and_save(
                [file_path],
                output_directory=output_dir,
                model_or_model_path=MODEL,
                save_model_outputs=False,
                save_notes=False,
                save_midi=True,
                sonify_midi=False,
            )
        else:
            # 🕹 Fallback for older library versions
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


def validate_model_load() -> bool:
    """Explicit model‑load test for diagnostics."""
    try:
        if os.path.exists(MODEL):
            logging.info("[✅] Platform‑native model path exists.")
            return True
        else:
            logging.warning("[⚠️] Model file does not exist — check BasicPitch installation.")
            return False
    except Exception as e:
        logging.error(f"[❌] Model load validation failed: {e}")
        return False