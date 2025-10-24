# filename: scripts/extract_features.py

import argparse
import os
import torch
from tqdm import tqdm
from utils.utils import (
    get_all_wav_files,
    separate_stems,
    extract_audio_features,
    extract_midi,
    setup_logging,
    log_performance,
    check_if_processed
)
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_track(wav_path, output_dir, device, log_file):
    track_name = wav_path.name
    try:
        if check_if_processed(wav_path, output_dir):
            print(f"[⏩] Skipping {track_name} — already processed with valid output.")
            return

        print(f"[▶️] {track_name}")

        # Step 1: Stem separation
        stems_dir = separate_stems(wav_path, output_dir, device)

        # Step 2: Feature extraction
        features = extract_audio_features(wav_path, stems_dir)

        # Step 3: MIDI extraction
        midi_path = extract_midi(wav_path, stems_dir, device)

        log_performance(wav_path, log_file, status=f"✅ success in {features['duration']:.2f}s")

    except Exception as e:
        log_performance(wav_path, log_file, status=f"error: {e}")
        print(f"[❌] {track_name} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="🎵 Extract features and MIDI from WAV files")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    device = "mps" if args.use_gpu and torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    wav_files_by_genre = get_all_wav_files(args.input_dir)

    for genre, wav_files in wav_files_by_genre.items():
        print(f"[🎵] Processing genre: {genre}")
        log_file = setup_logging(args.log_dir, genre)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            progress = tqdm(total=len(wav_files), ncols=120)
            for wav_path in wav_files:
                futures.append(executor.submit(process_track, wav_path, args.output_dir, device, log_file))
            for future in as_completed(futures):
                progress.update(1)
            progress.close()

if __name__ == "__main__":
    main()