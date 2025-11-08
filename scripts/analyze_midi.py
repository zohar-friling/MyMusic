# ============================================
# File: scripts/analyze_midi.py
# Stage 3 – MIDI Structural Analysis (flat output mode)
# Run from:  MyMusic/
# Output pattern: fusion_models/<genre>/<track_name>.analysis.json
#
# 🛠️ Added:
#  - Real-time console updates
#  - Genre-level and per-track performance logs
#  - Better handling of empty/missing folders
#  - Extensive comments for maintainability
# ============================================

import os
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pretty_midi
from music21 import converter, chord

# ---- Use Stage-2 utils if available, else fallback simple logger ----
try:
    from scripts.utils.utils import init_logger, log_performance
except Exception:
    def init_logger(*args, **kwargs):
        pass
    def log_performance(log_file, track_name, status, elapsed):
        """Fallback lightweight logger writing JSONL rows."""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "track": track_name,
            "status": status,
            "elapsed_sec": round(elapsed, 3),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

# ---------- Configuration ----------
ROOT = Path(__file__).resolve().parents[1]     # Points to MyMusic/
DATASET_FEATURES = ROOT / "dataset_features"
FUSION_MODELS = ROOT / "fusion_models"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"analyze_midi_{TIMESTAMP}.jsonl"

# ---------- Helpers ----------
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

def safe_name(name: str) -> str:
    """Sanitize filenames for cross-platform safety."""
    return SAFE_NAME_RE.sub("_", name.strip().replace(" ", "_"))

def detect_key(midi_path: Path) -> str:
    """Infer musical key using music21."""
    try:
        s = converter.parse(str(midi_path))
        k = s.analyze("key")
        return f"{k.tonic.name}_{k.mode}"
    except Exception:
        return "Unknown"

def extract_chords(midi_path: Path):
    """Extract coarse chord names from MIDI via chordify()."""
    try:
        s = converter.parse(str(midi_path))
        ch = s.chordify().recurse().getElementsByClass(chord.Chord)
        names, seen = [], set()
        for c in ch:
            n = str(c.commonName)
            if n not in seen:
                seen.add(n)
                names.append(n)
        return names
    except Exception:
        return []

def compute_polyphony(pm: pretty_midi.PrettyMIDI) -> float:
    """Average number of simultaneous notes over track duration."""
    spans = [(n.start, n.end) for inst in pm.instruments for n in inst.notes]
    if not spans:
        return 0.0
    spans.sort()
    events = [(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans]
    events.sort()
    active, last_t, area = 0, events[0][0], 0.0
    for t, delta in events:
        area += active * (t - last_t)
        active += delta
        last_t = t
    duration = max(1e-6, spans[-1][1] - spans[0][0])
    return round(area / duration, 3)

def motif_density(onsets):
    """Heuristic: proportion of short inter-onset intervals (density)."""
    if len(onsets) < 2:
        return 0.0
    diffs = np.diff(sorted(onsets))
    if not len(diffs):
        return 0.0
    med = np.median(diffs)
    return round(float((diffs < med).sum()) / len(diffs), 3)

def note_distribution(pm: pretty_midi.PrettyMIDI):
    """Percentage of notes in low/mid/high pitch ranges."""
    pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
    if not pitches:
        return {"low": 0, "mid": 0, "high": 0}
    total = len(pitches)
    low = sum(p < 60 for p in pitches)
    mid = sum(60 <= p < 72 for p in pitches)
    high = sum(p >= 72 for p in pitches)
    return {
        "low": round(100 * low / total, 1),
        "mid": round(100 * mid / total, 1),
        "high": round(100 * high / total, 1),
    }

# ---------- Core ----------
def analyze_track(mid_path: Path, json_path: Path, genre: str):
    """Analyze one MIDI+JSON pair and write an analysis.json."""
    start = time.time()
    track_name = safe_name(mid_path.stem)
    out_dir = FUSION_MODELS / genre
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{track_name}.analysis.json"

    # Skip if already processed
    if out_file.exists():
        log_performance(LOG_FILE, track_name, "skipped_exists", time.time() - start)
        return {"track": track_name, "status": "skipped"}

    try:
        # Load source MIDI and feature JSON
        pm = pretty_midi.PrettyMIDI(str(mid_path))
        feat = json.loads(Path(json_path).read_text())
        tempo = float(feat.get("tempo", 0.0))
        onsets = list(feat.get("onsets", []))

        # Derived musical features
        key_detected = detect_key(mid_path)
        chords = extract_chords(mid_path)
        poly = compute_polyphony(pm)
        motif = motif_density(onsets)
        dist = note_distribution(pm)
        duration = round(pm.get_end_time(), 3)

        # Naive sectioning (placeholder for ML-based segmentation)
        sections = ["single_section"]
        if duration >= 60:
            sections = ["intro", "main", "outro"]
        if duration >= 180:
            sections = ["intro", "A", "B", "bridge", "outro"]

        analysis = {
            "track_name": track_name,
            "genre_source": genre,
            "tempo_avg": round(tempo, 3),
            "key_detected": key_detected,
            "chords": chords,
            "polyphony_index": poly,
            "motif_density": motif,
            "note_distribution": dist,
            "sections": sections,
            "duration_sec": duration,
            "analyzed_at": TIMESTAMP,
            "sources": {
                "midi": str(mid_path),
                "features_json": str(json_path)
            }
        }

        out_file.write_text(json.dumps(analysis, indent=2))
        elapsed = time.time() - start
        log_performance(LOG_FILE, track_name, "success", elapsed)
        tqdm.write(f"✅ {genre}/{track_name} analyzed in {elapsed:.2f}s")
        return {"track": track_name, "status": "done", "elapsed": elapsed}

    except Exception as e:
        elapsed = time.time() - start
        log_performance(LOG_FILE, track_name, f"error: {e}", elapsed)
        tqdm.write(f"❌ {genre}/{track_name} failed ({e})")
        return {"track": track_name, "status": f"error: {e}", "elapsed": elapsed}


def process_genre(genre_path: Path):
    """Process all MIDIs inside one genre folder."""
    genre = genre_path.name
    mid_dir = genre_path / "mid"
    json_dir = genre_path / "json"
    if not mid_dir.exists() or not json_dir.exists():
        tqdm.write(f"⚠️  Missing directories for {genre}, skipping.")
        return []

    mids = list(mid_dir.glob("*.mid"))
    if not mids:
        tqdm.write(f"⚠️  No MIDI files found for {genre}.")
        return []

    tqdm.write(f"\n▶️  Processing genre: {genre} ({len(mids)} tracks)")
    results = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = []
        for mid_file in mids:
            json_file = json_dir / f"{mid_file.stem}.json"
            if not json_file.exists():
                log_performance(LOG_FILE, mid_file.stem, "missing_json", 0.0)
                tqdm.write(f"⚠️  Missing JSON for {mid_file.name}")
                continue
            futures.append(ex.submit(analyze_track, mid_file, json_file, genre))

        for f in tqdm(as_completed(futures), total=len(futures), desc=f"{genre}", leave=False):
            results.append(f.result())

    # Genre-level summary log entry
    done = sum(1 for r in results if r["status"].startswith("done"))
    skipped = sum(1 for r in results if "skipped" in r["status"])
    errors = sum(1 for r in results if "error" in r["status"])
    tqdm.write(f"✅ Finished {genre}: {done} done, {skipped} skipped, {errors} errors.")
    return results


def main():
    """Main controller for Stage 3 analysis."""
    print("🎼 Stage 3: MIDI Structural Analysis (flat outputs)")
    print(f"Logs → {LOG_FILE}\n")
    start = time.time()

    try:
        init_logger(LOG_FILE)
    except Exception:
        pass

    summary = {}
    genres = sorted([p for p in DATASET_FEATURES.iterdir() if p.is_dir()])
    total_genres = len(genres)
    if not total_genres:
        print("❌ No genre folders found in dataset_features/. Exiting.")
        return

    print(f"📦 Found {total_genres} genre folders.\n")

    for genre_path in genres:
        tqdm.write(f"--- Genre: {genre_path.name} ---")
        res = process_genre(genre_path)
        summary[genre_path.name] = res

    elapsed = round(time.time() - start, 2)
    print(f"\n🕒 Total runtime: {elapsed}s")
    print(f"📁 Outputs saved under fusion_models/<genre>/<track>.analysis.json")
    print(f"🧾 Logs written to {LOG_FILE}")

    # Write global summary JSON for post-run inspection
    with open(LOG_FILE.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()