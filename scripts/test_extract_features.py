# filename: scripts/test_extract_features.py
"""
🧪 Test: End-to-End Feature Extraction Pipeline (MyMusic)
✅ Verifies the complete chain:
    • Audio preprocessing
    • Onset + tempo extraction (librosa)
    • MIDI extraction via BasicPitch (ONNX auto-fallback)
    • Logging + feature persistence

🧠 Integrated with:
    - utils.utils (for model validation)
    - extract_features.ensure_basicpitch_model_ready()

💡 Behavior:
    - Automatically validates or downloads ONNX model before testing
    - Works on macOS / pyenv / ONNXRuntime / TensorFlow / CoreML setups
"""

import sys
from pathlib import Path

# 🧭 Add project root to import path (so utils and extract_features can be imported cleanly)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import shutil
import numpy as np
import soundfile as sf

# ✅ Import core pipeline utilities
from scripts.utils.utils import (
    extract_audio_features,
    extract_midi,
    save_json,
    is_audio_valid,
    setup_logging,
    validate_model_load,
)

# ✅ Import model bootstrapper from main pipeline
from scripts.extract_features import ensure_basicpitch_model_ready


# -----------------------------------------------------------
# 🧪 Test Directories and Files
# -----------------------------------------------------------
TEST_ROOT = Path("test_data")
TEST_AUDIO_DIR = TEST_ROOT / "dataset" / "test_genre"
TEST_OUT_DIR = TEST_ROOT / "dataset_features" / "test_genre"
TEST_LOG_DIR = TEST_ROOT / "logs"

TEST_WAV = TEST_AUDIO_DIR / "test_silence.wav"
FEATURES_JSON = TEST_OUT_DIR / "test_silence" / "audio_features.json"
MIDI_FILE = TEST_OUT_DIR / "test_silence" / "test_silence.mid"


# -----------------------------------------------------------
# 🧼 Environment Preparation
# -----------------------------------------------------------
def clean():
    """Clean all temporary test artifacts."""
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


# -----------------------------------------------------------
# 🎧 Create Dummy Audio File
# -----------------------------------------------------------
def create_test_wav():
    """Create a simple 2s silent WAV file for testing."""
    os.makedirs(TEST_AUDIO_DIR, exist_ok=True)
    sr = 22050
    duration = 2
    silence = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(TEST_WAV, silence, sr)
    print("[🎧] Created test WAV:", TEST_WAV)


# -----------------------------------------------------------
# 🚀 Execute End-to-End Pipeline
# -----------------------------------------------------------
def run_pipeline():
    """Run the feature extraction sequence."""
    try:
        print("[⚙️] Checking BasicPitch model...")
        ensure_basicpitch_model_ready()

        # 🔍 Validate model load once before processing
        if not validate_model_load():
            print("[⚠️] Warning: Model validation failed, will attempt fallback...")

        print("[🚀] Running pipeline...")
        log_file = setup_logging(str(TEST_LOG_DIR))
        print("[🔍] Log file:", log_file)

        # 🧪 Input sanity check
        assert is_audio_valid(str(TEST_WAV)), "Invalid or unreadable WAV file"
        print("[🎵] Found 1 WAV file")

        # 🎼 MIDI extraction (auto ONNX model)
        print(f"Predicting MIDI for {TEST_WAV}...")
        midi_ok = extract_midi(str(TEST_WAV), str(TEST_OUT_DIR / "test_silence"))
        assert midi_ok, "MIDI extraction failed"

        # 🎚 Audio feature extraction
        features = extract_audio_features(str(TEST_WAV))
        assert features is not None, "Failed to extract audio features"
        save_json(features, str(FEATURES_JSON))

        print("[✅] Features saved:", FEATURES_JSON)
        return True

    except Exception as e:
        print("[❌] Pipeline failed:", e)
        return False


# -----------------------------------------------------------
# ✅ Output Assertions
# -----------------------------------------------------------
def assert_outputs():
    """Ensure all expected files exist."""
    assert FEATURES_JSON.exists(), "❌ Features JSON not created"
    assert MIDI_FILE.exists(), "❌ MIDI file not created"
    assert_log_file()
    print("[📝] Log file:", list(TEST_LOG_DIR.glob('run_*.log'))[0])
    print("[📊] Summary file:", TEST_LOG_DIR / "performance_summary.log")


def assert_log_file():
    """Ensure the log file was generated."""
    logs = list(TEST_LOG_DIR.glob("run_*.log"))
    assert logs, "❌ No run_*.log file found"


# -----------------------------------------------------------
# 🧪 Main Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    clean()
    create_test_wav()
    assert run_pipeline(), "Pipeline run failed"
    assert_outputs()
    print("\n🎉 All tests passed successfully.")
    clean()