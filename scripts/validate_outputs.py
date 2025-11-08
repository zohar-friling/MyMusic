# filename: scripts/validate_outputs.py
"""
Validation script for checking quality of extracted features:
✅ MIDI: at least 5 note events
✅ JSON: tempo within 60–180 BPM and at least 5 onsets
"""

import os
from pathlib import Path
import json
from pretty_midi import PrettyMIDI

def is_high_quality_midi(midi_path):
    try:
        midi = PrettyMIDI(str(midi_path))
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        return total_notes >= 5
    except Exception:
        return False

def is_high_quality_json(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        tempo = data.get("tempo", 0)
        onsets = data.get("onsets", [])
        return (60 <= tempo <= 180) and len(onsets) >= 5
    except Exception:
        return False

def scan_directory(root="dataset_features"):
    root_path = Path(root)
    results = []
    for genre_dir in root_path.iterdir():
        if genre_dir.is_dir():
            for track_dir in genre_dir.iterdir():
                if track_dir.is_dir():
                    midi_file = track_dir / (track_dir.name + "_basic_pitch.mid")
                    json_file = track_dir / "audio_features.json"
                    midi_ok = is_high_quality_midi(midi_file) if midi_file.exists() else False
                    json_ok = is_high_quality_json(json_file) if json_file.exists() else False
                    results.append((str(track_dir), midi_ok, json_ok))
    return results

if __name__ == "__main__":
    print("🔍 Validating output files in dataset_features/ ...\n")
    summary = scan_directory()
    for track_path, midi_ok, json_ok in summary:
        status = []
        status.append("✅ MIDI" if midi_ok else "❌ MIDI")
        status.append("✅ JSON" if json_ok else "❌ JSON")
        print(f"{track_path}: {' | '.join(status)}")
