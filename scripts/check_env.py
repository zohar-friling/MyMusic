import importlib
import subprocess
import sys
import os
import platform

REQUIRED_PACKAGES = [
    "torch",
    "torchaudio",
    "demucs",
    "basic_pitch",
    "librosa",
    "soundfile",
    "tqdm",
    "scipy",
    "numpy",
]

def check_gpu_support():
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        print(f"✅ Torch version: {torch.__version__}")
        print(f"   🧠 MPS GPU Available: {mps_available}")
    except Exception as e:
        print(f"❌ Torch check failed: {e}")

def check_package(package):
    try:
        module = importlib.import_module(package)
        print(f"✅ {package} is installed ({module.__version__ if hasattr(module, '__version__') else 'no __version__'})")
    except ImportError:
        print(f"❌ {package} is NOT installed")

def check_command_line_tool(tool):
    result = subprocess.run(['which', tool], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Command-line tool '{tool}' is installed at: {result.stdout.strip()}")
    else:
        print(f"❌ Command-line tool '{tool}' is NOT found")

def main():
    print("🎛️ System Info:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.executable}")
    print("")

    print("🔍 Checking Python packages:")
    for pkg in REQUIRED_PACKAGES:
        check_package(pkg)

    print("\n⚙️ Checking CLI tools:")
    check_command_line_tool("demucs")

    print("\n🧠 GPU (MPS) Support:")
    check_gpu_support()

if __name__ == "__main__":
    main()