# filename: scripts/check_env.py
"""
Check that required libraries and system modules are available.
✅ Confirms proper environment for BasicPitch (CoreML or ONNX) + Librosa + Torch.
"""

import sys, importlib, platform

REQUIRED = [
    "librosa", "basic_pitch", "soundfile",
    "numpy", "scipy", "torch", "pretty_midi", "resampy"
]

print("🔍 Checking environment:\n")
for m in REQUIRED:
    try:
        importlib.import_module(m)
        print(f"[✅] {m}")
    except ImportError:
        print(f"[❌] {m} missing")

print("\n🔧 System Info:")
print(f"  Python  : {sys.version.split()[0]}")
print(f"  Platform: {platform.system()} {platform.machine()}")
print(f"  Prefix  : {sys.prefix}")

# Check CoreML or ONNX availability
try:
    import coremltools
    print("[✅] CoreML backend available (macOS optimized)")
except ImportError:
    print("[ℹ️] CoreML not found – BasicPitch will use ONNX or TF backend instead.")

try:
    import lzma
    print("[✅] lzma is available in stdlib")
except ImportError:
    print("[❌] lzma missing – recompile Python with '--with-lzma'")