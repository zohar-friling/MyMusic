import os
import subprocess
from pathlib import Path

# 🎯 Your root dataset folder
ROOT_DIR = Path("/Users/zohar/Library/Mobile Documents/com~apple~CloudDocs/MyMusic/dataset")

# 🎧 Target format settings
TARGET_SR = 44100
TARGET_FMT = "pcm_f32le"  # 32-bit float

def convert_mp3_to_wav(root: Path):
    print(f"[START] Scanning and converting .mp3 files in: {root}\n")
    print(f"{'Genre':<12} {'MP3':>5} {'Converted':>10} {'Skipped':>8} {'Reconverted':>12} {'Errors':>7}")
    print("=" * 60)

    for genre_dir in sorted(root.iterdir()):
        if not genre_dir.is_dir():
            continue

        mp3_dir = genre_dir / "mp3"
        wav_dir = genre_dir / "wav"
        if not mp3_dir.exists():
            continue
        wav_dir.mkdir(exist_ok=True)

        mp3_files = sorted(mp3_dir.glob("*.mp3"))
        converted = 0
        skipped = 0
        reconverted = 0
        errors = 0

        for mp3_file in mp3_files:
            wav_file = wav_dir / (mp3_file.stem + ".wav")

            if wav_file.exists():
                if wav_file.stat().st_size == 0:
                    # Zero-byte WAV — reconvert
                    reconverted += 1
                else:
                    skipped += 1
                    continue

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(mp3_file),
                "-ar", str(TARGET_SR),
                "-ac", "2",
                "-c:a", "pcm_f32le",  # ✅ explicitly set to float WAV encoder
                str(wav_file)
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not wav_file.exists() or wav_file.stat().st_size == 0:
                print(f"  ❌ Conversion failed: {mp3_file.name}")
                print(result.stderr.decode())
                errors += 1
                continue

            if reconverted:
                print(f"  🔁 Reconverted: {mp3_file.name}")
            else:
                converted += 1
                print(f"  ✅ Converted: {mp3_file.name}")

        print(f"{genre_dir.name:<12} {len(mp3_files):>5} {converted:>10} {skipped:>8} {reconverted:>12} {errors:>7}")

    print("\n✅ Done. You can rerun this script safely anytime.\n")

if __name__ == "__main__":
    convert_mp3_to_wav(ROOT_DIR)