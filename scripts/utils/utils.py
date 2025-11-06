# filename: scripts/utils/utils.py
"""
Utility functions for the MyMusic feature extraction pipeline.
✅ Fully compatible with BasicPitch 0.4.x (Python 3.10–3.11 / macOS ARM64 + CoreML backend)

🧠 FINAL PATCH – NOV 2025
────────────────────────────
This version removes all references to missing BasicPitch constants
(e.g., ICASSP_2022_MODEL_PATH) and dynamically resolves the model path.

🪄 Behavior summary:
  • macOS → CoreML (.mlmodel)
  • Linux/Windows → ONNX (.onnx)
  • Automatically detects the installed path under basic_pitch/saved_models/
  • No hardcoded or external downloads required.
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

# ✅ Import only the stable API function
from basic_pitch.inference import predict_and_save


# -----------------------------------------------------------
# 🧭 Model path auto‑resolution
# -----------------------------------------------------------
def resolve_basicpitch_model() -> str:
    """
    Dynamically locate the BasicPitch model inside the installed package.
    Falls back gracefully if path cannot be found.

    Returns:
        str: Path to the CoreML (.mlmodel) or ONNX (.onnx) model file
    """
    try:
        import basic_pitch
        root = os.path.dirname(basic_pitch.__file__)
        # macOS (Darwin) uses CoreML; other OSes use ONNX
        if platform.system() == "Darwin":
            model_path = os.path.join(root, "saved_models/icassp_2022/model.mlmodel")
        else:
            model_path = os.path.join(root, "saved_models/icassp_2022/model.onnx")

        if not os.path.exists(model_path):
            logging.warning(f"[⚠️] Expected model not found at {model_path}")
        else:
            logging.info(f"[✅] Using BasicPitch model: {model_path}")
        return model_path

    except Exception as e:
        logging.error(f"[❌] Failed to resolve BasicPitch model path: {e}")
        return None


# ✅ Global model constant
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
    """
    Validate that an audio file exists, is readable, and non-empty.
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
    Extract basic rhythmic and onset features from an audio file.
    Returns tempo (BPM) and onset timings (seconds).
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
    Extract MIDI from an audio file using BasicPitch.
    Uses platform‑appropriate model path.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        sig = inspect.signature(predict_and_save)

        # ✅ Modern API with explicit model path
        if "model_or_model_path" in sig.parameters:
            logging.info(f"[🎹] Running BasicPitch on {os.path.basename(file_path)}")
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
            # 🕹 Legacy API fallback (rare)
            logging.info(f"[🎹] Legacy BasicPitch mode for {os.path.basename(file_path)}")
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
    """Persist extracted feature data to disk."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"[❌] Failed to save JSON: {e}")


def log_performance(track_name: str, start_time: datetime, end_time: datetime, status: str, log_dir: str):
    """Append per‑track timing and status to performance log."""
    try:
        duration = (end_time - start_time).total_seconds()
        with open(os.path.join(log_dir, "performance_summary.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} - {track_name} - {status} ({duration:.2f}s)\n")
    except Exception as e:
        logging.error(f"[❌] Failed to log performance for {track_name}: {e}")


# -----------------------------------------------------------
# 🧪 Model validation
# -----------------------------------------------------------
def validate_model_load() -> bool:
    """Check that the resolved BasicPitch model file actually exists."""
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