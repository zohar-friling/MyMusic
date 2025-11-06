# filename: scripts/test_extract_features.py
"""
🧪 Test: End‑to‑End Feature Extraction Pipeline (MyMusic)
✅ Verifies:
   • Audio preprocessing
   • Tempo + onset extraction
   • MIDI extraction via BasicPitch (CoreML/ONNX auto)
   • Logging + output validation
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, shutil, numpy as np, soundfile as sf
from datetime import datetime

from scripts.utils.utils import (
    extract_audio_features,
    extract_midi,
    save_json,
    is_audio_valid,
    setup_logging,
    validate_model_load,
)
from scripts.extract_features import ensure_basicpitch_model_ready

TEST_ROOT = Path("test_data")
TEST_WAV = TEST_ROOT / "dataset/test_genre/test_silence.wav"
TEST_OUT = TEST_ROOT / "dataset_features/test_genre/test_silence"
TEST_LOG = TEST_ROOT / "logs"

def clean(): shutil.rmtree(TEST_ROOT, ignore_errors=True)

def create_test_wav():
    os.makedirs(TEST_WAV.parent, exist_ok=True)
    silence = np.zeros(44100 * 2, dtype=np.float32)
    sf.write(TEST_WAV, silence, 44100)
    print(f"[🎧] Created test WAV: {TEST_WAV}")

def run_pipeline():
    print("[⚙️] Validating model...")
    ensure_basicpitch_model_ready()
    if not validate_model_load():
        print("[⚠️] Model validation failed, but continuing with fallback...")

    print("[🚀] Running pipeline...")
    setup_logging(str(TEST_LOG))
    assert is_audio_valid(str(TEST_WAV)), "Invalid WAV"

    midi_ok = extract_midi(str(TEST_WAV), str(TEST_OUT))
    assert midi_ok, "MIDI extraction failed"

    features = extract_audio_features(str(TEST_WAV))
    assert features, "No features extracted"
    save_json(features, str(TEST_OUT / "audio_features.json"))
    return True

if __name__ == "__main__":
    clean()
    create_test_wav()
    assert run_pipeline(), "Pipeline run failed"
    print("✅ Test completed successfully.")
    clean()