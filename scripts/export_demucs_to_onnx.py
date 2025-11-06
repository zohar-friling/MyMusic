# NOT WORKING 
import torch
import torch.nn as nn
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pathlib import Path

class DemucsWrapper(nn.Module):
    """
    Wrapper class to make Demucs exportable to ONNX.
    It exposes a standard forward() method compatible with torch.onnx.export().
    """
    def __init__(self, model_name="htdemucs"):
        super().__init__()
        print(f"[INFO] Loading pretrained Demucs model: {model_name}")
        self.model = get_model(model_name).cpu()
        self.model.eval()

    def forward(self, audio: torch.Tensor):
        """
        audio: shape (batch, channels, samples)
        """
        # The internal apply_model handles multi‑stem inference
        sources = apply_model(self.model, audio, device="cpu")[0]
        return sources

# ──────────────────────────────────────────────────────────────
# EXPORT LOGIC
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_NAME = "htdemucs"
    OUTPUT_PATH = Path("htdemucs.onnx")

    # Initialize wrapper
    model = DemucsWrapper(MODEL_NAME)

    # Dummy stereo audio: 10 seconds @ 44.1 kHz
    SAMPLE_RATE = 44100
    DURATION_SEC = 10
    dummy_audio = torch.randn(1, 2, SAMPLE_RATE * DURATION_SEC)

    print(f"[INFO] Exporting {MODEL_NAME} to ONNX → {OUTPUT_PATH}")
    torch.onnx.export(
        model, 
        dummy_audio, 
        OUTPUT_PATH,
        input_names=["audio"],
        output_names=["stems"],
        opset_version=11,
        dynamic_axes={"audio": {2: "samples"}}
    )

    print(f"[✅] Export complete → {OUTPUT_PATH.resolve()}")
