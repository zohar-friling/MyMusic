# filename: scripts/utils/utils.py
"""
Utility functions for the MyMusic feature extraction pipeline.
✅ Fully compatible with BasicPitch 0.4.x (Python 3.10–3.11 / macOS ARM64 + CoreML backend)

🧠 FINAL PATCH – NOV 2025
────────────────────────────
This version:
  • Supports `.mlpackage` model bundles required by CoreML
  • Fixes model resolution issues on macOS
  • Fully backward-compatible with ONNX or legacy paths

🪄 Behavior summary:
  • macOS → CoreML (.mlpackage)
  • Linux/Windows → ONNX (.onnx)
  • Dynamically resolves paths under basic_pitch/saved_models/
"""

import os
import platform
import json
import logging
import librosa
import soundfile as sf
import numpy as np
import inspect
from datetime import datetime

from basic_pitch.inference import predict_and_save


# -----------------------------------------------------------
# 🧭 Model path auto‑resolution (💡 Updated to support .mlpackage bundles)
# -----------------------------------------------------------
def resolve_basicpitch_model() -> str:
    """
    Dynamically locate the BasicPitch model inside the installed package.
    Returns the path to the CoreML (.mlpackage) or ONNX (.onnx) model file.
    """
    try:
        import basic_pitch
        root = os.path.dirname(basic_pitch.__file__)

        if platform.system() == "Darwin":
            # ✅ macOS: CoreML expects the full .mlpackage directory, not a file inside it
            candidate_paths = [
                os.path.join(root, "saved_models/icassp_2022/nmp.mlpackage"),  # ✅ correct modern path (directory)
                os.path.join(root, "saved_models/icassp_2022/nmp.mlpackage/Data/com.apple.CoreML/model.mlmodel"),  # legacy fallback
                os.path.join(root, "saved_models/icassp_2022/model.mlmodel"),  # legacy fallback
            ]
        else:
            # ✅ Linux/Windows: ONNX preferred
            candidate_paths = [
                os.path.join(root, "saved_models/icassp_2022/nmp.onnx"),
                os.path.join(root, "saved_models/icassp_2022/model.onnx"),
            ]

        for path in candidate_paths:
            if os.path.exists(path):
                logging.info(f"[✅] Using BasicPitch model: {path}")
                return path

        logging.warning(f"[⚠️] No valid BasicPitch model found in: {candidate_paths}")
        return None

    except Exception as e:
        logging.error(f"[❌] Failed to resolve BasicPitch model path: {e}")
        return None


# ✅ Global model path (resolved at load time)
MODEL = resolve_basicpitch_model()


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
    logging.getLogger().addHandler(logging.StreamHandler())
    return log_file


# -----------------------------------------------------------
# 🎧 Audio validation
# -----------------------------------------------------------
def is_audio_valid(filepath: str) -> bool:
    """Check if an audio file exists, is non-empty, and can be opened."""
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
    Extract tempo (BPM) and onset timings (seconds) from audio.
    Returns a dict with 'tempo' and 'onsets'.
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
# 🎼 MIDI extraction (BasicPitch, model-aware)
# -----------------------------------------------------------
def extract_midi(file_path: str, output_dir: str) -> bool:
    """
    Run BasicPitch inference and extract a MIDI file.
    Selects model automatically (CoreML or ONNX).
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        sig = inspect.signature(predict_and_save)

        # ✅ Modern BasicPitch with model path param
        if "model_or_model_path" in sig.parameters:
            logging.info(f"[🎹] Extracting MIDI from: {os.path.basename(file_path)}")
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
            # 🕹 Legacy fallback
            logging.info(f"[🎹] Using legacy BasicPitch for: {os.path.basename(file_path)}")
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
# 💾 JSON and performance helpers
# -----------------------------------------------------------
def save_json(data: dict, output_path: str):
    """Write a dictionary to JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"[❌] Failed to save JSON to {output_path}: {e}")


def log_performance(track_name: str, start_time: datetime, end_time: datetime, status: str, log_dir: str):
    """Write performance metrics to log file."""
    try:
        duration = (end_time - start_time).total_seconds()
        with open(os.path.join(log_dir, "performance_summary.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} - {track_name} - {status} ({duration:.2f}s)\n")
    except Exception as e:
        logging.error(f"[❌] Failed to log performance: {e}")


# -----------------------------------------------------------
# 🧪 Model validation
# -----------------------------------------------------------
def validate_model_load() -> bool:
    """Verify that the resolved model path exists on disk."""
    try:
        if MODEL and os.path.exists(MODEL):
            logging.info(f"[✅] BasicPitch model verified: {MODEL}")
            return True
        else:
            logging.warning("[⚠️] BasicPitch model missing or unresolved.")
            return False
    except Exception as e:
        logging.error(f"[❌] Model validation failed: {e}")
        return False