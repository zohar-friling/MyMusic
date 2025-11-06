# filename: scripts/check_env.py

"""
Check that required libraries and system modules are available.
Specifically confirms lzma support, critical for proper audio/MIDI processing.
"""

import sys
import importlib
import platform

REQUIRED_MODULES = [
    "librosa",
    "basic_pitch",
    "lzma",
    "soundfile",
    "numpy",
    "scipy",
    "torch",
    "pretty_midi",
    "resampy"
]

print("🔍 Checking environment for required modules:\n")

for module in REQUIRED_MODULES:
    try:
        importlib.import_module(module)
        print(f"[✅] {module}")
    except ImportError:
        print(f"[❌] {module} is MISSING")

# Additional test for lzma availability at system level
print("\n🔧 Python build info:")
print(f"  Python version  : {sys.version.split()[0]}")
print(f"  Platform         : {platform.system()} {platform.machine()}")
print(f"  Pyenv prefix     : {sys.prefix}")

try:
    import lzma
    print("\n[✅] lzma is correctly available via stdlib")
except ImportError as e:
    print("\n[❌] lzma is missing — this likely means your pyenv Python was compiled WITHOUT lzma/xz support.")
    print("    ➤ Rebuild Python with:")
    print('      PYTHON_CONFIGURE_OPTS="--with-lzma" pyenv install 3.10.x')