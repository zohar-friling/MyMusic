# filename: scripts/utils/utils.py
"""
Utility functions for the MyMusic feature extraction pipeline.
✅ Compatible with BasicPitch 0.4.x (Python 3.10–3.11 / macOS ARM64, CoreML backend)

🧠 FINALIZED VERSION – NOV 2025
────────────────────────────
This version removes all references to the deprecated `load_model()`
(which no longer exists in current BasicPitch builds) and uses
BasicPitch’s official constant `ICASSP_2022_MODEL_PATH`.

🪄 Behavior summary:
  • macOS automatically uses CoreML (.mlmodel)
  • Linux/Windows automatically use ONNX (.onnx)
  • No manual download or cache needed
  • No dead HuggingFace links or missing model errors
"""

import os
import json
import logging
import librosa
import soundfile as sf
import numpy as np
import inspect
from datetime import datetime

# ✅ Import the stable BasicPitch API
# - predict_and_save(): main inference method
# - ICASSP_2022_MODEL_PATH: correct model file for this platform
from basic_pitch.inference import predict_and_save
from basic_pitch.models import ICASSP_2022_MODEL_PATH

# ✅ Core model path (resolves to .mlmodel on macOS or .onnx elsewhere)
MODEL = ICASSP_2022_MODEL_PATH


# -----------------------------------------------------------
# ⚙️ Logging setup
# -----------------------------------------------------------
def setup_logging(log_dir: str) -> str:
    """Create log directory + file and configure logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    open(log_file, "a").close()
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # StreamHandler lets you see logs in the console too
    logging.getLogger().addHandler(logging.StreamHandler())
    return log_file


# -----------------------------------------------------------
# 🎧 Audio validation
# -----------------------------------------------------------
def is_audio_valid(filepath: str) -> bool:
    """
    Validate that an audio file exists, is readable, and non-empty.

    Returns:
        bool: True if audio file is valid.
    """
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
    """
    Extract simple rhythmic and onset-based features using Librosa.

    Returns:
        dict: Contains 'tempo' (float) and 'onsets' (list of seconds)
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return {"tempo": tempo, "onsets": onset_times.tolist()}
    except Exception as e:
        logging.error(f"[❌] Failed to extract audio features from {file_path}: {e}")
        return None


# -----------------------------------------------------------
# 🎼 MIDI extraction (BasicPitch, platform‑aware)
# -----------------------------------------------------------
def extract_midi(file_path: str, output_dir: str) -> bool:
    """
    Extract a MIDI file from an audio clip using BasicPitch.
    Uses the correct model automatically (CoreML or ONNX).

    Returns:
        bool: True if successful.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        sig = inspect.signature(predict_and_save)

        if "model_or_model_path" in sig.parameters:
            logging.info(f"[🎹] Using BasicPitch model for {os.path.basename(file_path)}")
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
            # 🧩 Fallback for older library versions (rare)
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
        logging.error(f"[❌] MIDI extraction failed for {file_path}: {e}")
        return False


# -----------------------------------------------------------
# 💾 JSON & performance helpers
# -----------------------------------------------------------
def save_json(data: dict, output_path: str):
    """Save feature data as JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"[❌] Failed to save JSON: {e}")


def log_performance(track_name: str, start_time: datetime, end_time: datetime, status: str, log_dir: str):
    """Append per‑track timing and status to a log file."""
    try:
        duration = (end_time - start_time).total_seconds()
        with open(os.path.join(log_dir, "performance_summary.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} - {track_name} - {status} ({duration:.2f}s)\n")
    except Exception as e:
        logging.error(f"[❌] Failed to log performance for {track_name}: {e}")


# -----------------------------------------------------------
# 🧪 Model availability check
# -----------------------------------------------------------
def validate_model_load() -> bool:
    """Check if the BasicPitch model file exists."""
    try:
        if os.path.exists(MODEL):
            logging.info(f"[✅] BasicPitch model found at {MODEL}")
            return True
        else:
            logging.warning("[⚠️] Model file missing – check your installation.")
            return False
    except Exception as e:
        logging.error(f"[❌] Model validation failed: {e}")
        return False