import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pathlib import Path

def separate_stems(wav_path: Path, output_dir: Path, model_name="htdemucs", device=None):
    """
    Separates stems from a WAV file using Demucs with GPU fallback.
    
    Args:
        wav_path (Path): Path to the input WAV file.
        output_dir (Path): Path to the output directory for saving stems.
        model_name (str): Demucs model name (default: htdemucs).
        device (str|None): 'cuda', 'mps', 'cpu' or None (auto-select).
    
    Returns:
        stems_saved_paths (list[Path]) or raises Exception on failure.
    """
    try:
        # Prefer GPU
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        model = get_model(name=model_name).to(device)
        wav_tensor, sr = torchaudio.load(str(wav_path))
        
        if wav_tensor.shape[0] == 1:
            wav_tensor = wav_tensor.repeat(2, 1)  # Duplicate mono to stereo
        wav_tensor = wav_tensor.to(device)

        with torch.no_grad():
            sources = apply_model(model, wav_tensor, device=device)[0]

        # Output directory: output_dir/<genre>/stems/<track_name>/
        track_name = wav_path.stem
        track_stem_dir = output_dir / "stems" / track_name
        track_stem_dir.mkdir(parents=True, exist_ok=True)

        stems_paths = []
        for i, stem in enumerate(model.sources):
            stem_waveform = sources[i].cpu()
            stem_path = track_stem_dir / f"{stem}.wav"
            torchaudio.save(str(stem_path), stem_waveform, sr)
            stems_paths.append(stem_path)

        return stems_paths

    except Exception as e:
        # If error on GPU, retry on CPU
        if device != "cpu":
            print(f"[WARN] GPU failed with: {e}. Falling back to CPU.")
            return separate_stems(wav_path, output_dir, model_name=model_name, device="cpu")
        else:
            raise RuntimeError(f"[FAIL] Failed on CPU as well: {e}")