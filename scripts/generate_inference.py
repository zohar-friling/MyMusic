#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 — First-hop Inference (Classical → Jazz)
-------------------------------------------------
Generates multiple "Jazz-style" latent variants of one classical track.

Architecture
------------
Encoder : 4 → 128 → 256  
Translator : 266 → 512 → 512 → 256  
Decoder : 256 → 128 → 4   (proxy feature space)

Input composition (266):
 256 latent + 4 normalized features (duration, motif_density, polyphony_index, tempo_avg)
 + 6 padding zeros (for missing conditioning dimensions in Stage 4)

Usage
-----
python3 scripts/generate_inference.py \
 --track 10_Mass_in_B_Minor__BWV_232__Qui_Sedes_Ad_Dexteram_Patris \
 --grid small --decode-now
"""

import argparse, json, sys, time, random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm

# ------------------------------------------------------------------
# 📁 Paths and constants
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "fusion_models" / "cache"
MODEL_PATH = PROJECT_ROOT / "fusion_models" / "fusion_core.pt"
DATASET_JSONL = CACHE_DIR / "fusion_dataset.jsonl"
SCHEMA_JSON = CACHE_DIR / "fusion_schema.json"
INFERENCE_ROOT = PROJECT_ROOT / "fusion_inference"
FUSION_CHAIN = ["classical","jazz","hijaz","salsa","trance"]

# ------------------------------------------------------------------
# 🧩 Utility functions
# ------------------------------------------------------------------
def utc_now(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
def load_json(p:Path): 
    with p.open("r",encoding="utf-8") as f: return json.load(f)
def append_jsonl(p:Path,obj:Dict[str,Any]):
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(obj)+"\n")
def load_dataset_entry(track_id:str)->Dict[str,Any]:
    with DATASET_JSONL.open("r",encoding="utf-8") as f:
        for line in f:
            try: row=json.loads(line)
            except: continue
            if row.get("track_id")==track_id: return row
    raise KeyError(f"Track {track_id} not found")
def safe_cosine(a,b):
    denom=(np.linalg.norm(a)*np.linalg.norm(b))+1e-9
    return float(np.dot(a,b)/denom)

# ------------------------------------------------------------------
# 🧠 FusionModel wrapper
# ------------------------------------------------------------------
class FusionModel:
    """Load fusion_core.pt and perform encode / translate / decode."""
    def __init__(self,path:Path,device="mps"):
        self.device=torch.device(
            device if torch.backends.mps.is_available() and device=="mps"
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Safe load (checkpoint unpickled only from trusted source)
        ckpt=torch.load(path,map_location="cpu")
        self.genre_order=ckpt.get("genre_order",FUSION_CHAIN)
        latent_dim=int(ckpt.get("latent_dim",256))
        input_dim=int(ckpt.get("input_dim",4))

        # Encoder (4 → 128 → 256)
        self.encoder=torch.nn.Sequential(
            torch.nn.Linear(input_dim,128),torch.nn.ReLU(),
            torch.nn.Linear(128,latent_dim))

        # ✅ Translator (266 → 512 → 512 → 256) matches Stage 4
        self.translator=torch.nn.Sequential(
            torch.nn.Linear(266,512),torch.nn.ReLU(),
            torch.nn.Linear(512,512),torch.nn.ReLU(),
            torch.nn.Linear(512,latent_dim))

        # Decoder (proxy features)
        self.decoder=torch.nn.Sequential(
            torch.nn.Linear(latent_dim,128),torch.nn.ReLU(),
            torch.nn.Linear(128,input_dim))

        # --- Load weights (lenient) ---
        if "encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder_state_dict"],strict=False)

        if "translator_state_dict" in ckpt:
            t_sd=ckpt["translator_state_dict"]
            # Handle 'fc.' prefix
            if all(k.startswith("fc.") for k in t_sd.keys()):
                t_sd={k.replace("fc.",""):v for k,v in t_sd.items()}
            try: self.translator.load_state_dict(t_sd,strict=False)
            except Exception as e: print(f"[WARN] Translator mismatch: {e}")

        self.decoder_available="decoder_state_dict" in ckpt
        if self.decoder_available:
            try:self.decoder.load_state_dict(ckpt["decoder_state_dict"],strict=False)
            except Exception as e:
                self.decoder_available=False
                print(f"[WARN] Decoder mismatch: {e}")

        for m in (self.encoder,self.translator,self.decoder):
            m.to(self.device).eval()
        self.model_hash=ckpt.get("model_hash",None)

    # --------------------------------------------------------------
    def encode(self,x):
        """Encode 4-feature vector → 256-latent and store features for translator."""
        self.last_features=x # retain for conditioning
        x=torch.from_numpy(x.astype(np.float32)).to(self.device)
        with torch.no_grad(): z=self.encoder(x)
        return z.cpu().numpy()

    def translate(self,z,src_idx,tgt_idx,sigma,beta,seed):
        """
        Translate latent → target genre latent.
        Concatenate latent(256) + 4 features + 6 zeros = 266 inputs.
        """
        seed_everything(seed)
        z=torch.from_numpy(z.astype(np.float32)).to(self.device)

        # Build conditioning vector
        feats=torch.tensor(getattr(self,"last_features",
                     torch.zeros(4,device=self.device)),
                     dtype=torch.float32,device=self.device)
        if feats.numel()<4:
            feats=torch.cat([feats,torch.zeros(4-feats.numel(),device=self.device)])
        pad=torch.zeros(6,device=self.device)  # missing aux conditioning
        inp=torch.cat([z,feats[:4],pad])  # → (266,)

        # Forward through translator
        with torch.no_grad(): z_t=self.translator(inp)
        eps=torch.randn_like(z_t)*float(sigma)
        z_out=(1-beta)*z_t+eps
        return z_out.cpu().numpy()

    def decode_features_proxy(self,z,temp):
        """Proxy decode latent → feature space for QC (JSON only)."""
        if not self.decoder_available:
            raise RuntimeError("Decoder missing in fusion_core.pt")
        z=torch.from_numpy(z.astype(np.float32)).to(self.device)
        with torch.no_grad(): xh=self.decoder(z*float(temp))
        return xh.cpu().numpy()

# ------------------------------------------------------------------
# 🎚️ Parameter grids
# ------------------------------------------------------------------
PRESETS={
 "small":{"sigma":[0.05,0.10],"beta":[0.8,1.0],
          "temperature":[0.9,1.0],"seeds":[111]},
 "medium":{"sigma":[0.05,0.10],"beta":[0.8,1.0],
           "temperature":[0.9,1.0],"seeds":[111,222,333]}
}
def build_param_grid(args):
    preset=PRESETS.get(args.grid,PRESETS["small"])
    sigma=args.sigma or preset["sigma"]
    beta=args.beta or preset["beta"]
    temp=args.temperature or preset["temperature"]
    seeds=args.seeds or preset["seeds"]
    return[{"sigma":s,"beta":b,"temperature":t,"seed":sd}
           for s in sigma for b in beta for t in temp for sd in seeds]

# ------------------------------------------------------------------
# 🚀 Main execution
# ------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser(description="Stage 5 Classical→Jazz inference")
    ap.add_argument("--track",required=True)
    ap.add_argument("--grid",choices=list(PRESETS.keys()),default="medium")
    ap.add_argument("--sigma",type=float,nargs="*")
    ap.add_argument("--beta",type=float,nargs="*")
    ap.add_argument("--temperature",type=float,nargs="*")
    ap.add_argument("--seeds",type=int,nargs="*")
    ap.add_argument("--decode-now",action="store_true")
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()

    if not MODEL_PATH.exists(): sys.exit(f"[FATAL] Missing model {MODEL_PATH}")
    if not DATASET_JSONL.exists(): sys.exit(f"[FATAL] Missing dataset {DATASET_JSONL}")

    row=load_dataset_entry(args.track)
    x_vec=row.get("x",{})
    schema=load_json(SCHEMA_JSON) if SCHEMA_JSON.exists() else {}
    feat_names=schema.get("feature_names",list(x_vec.keys()))
    x_np=np.array([float(x_vec[k]) for k in feat_names],np.float32)

    # Prepare output dirs
    track_id=row["track_id"]
    step_dir=INFERENCE_ROOT/track_id/"classical_to_jazz"
    ensure_dir(step_dir); ensure_dir(INFERENCE_ROOT/"logs")
    manifest_path=INFERENCE_ROOT/track_id/"manifest.json"
    batch_log=INFERENCE_ROOT/"logs"/f"stage5_{track_id}_c2j_{int(time.time())}.jsonl"

    model=FusionModel(MODEL_PATH)
    src_idx=model.genre_order.index("classical")
    tgt_idx=model.genre_order.index("jazz")

    z0=model.encode(x_np)
    grid=build_param_grid(args)
    print(f"[INFO] Device={model.device}, planned versions={len(grid)}")

    manifest={"track_id":track_id,"created":utc_now(),
              "model_hash":model.model_hash,
              "steps":{"classical_to_jazz":[]}}

    # Main loop
    for i,prm in enumerate(tqdm(grid,desc="Classical→Jazz"),start=1):
        vdir=step_dir/f"version_{i:02d}"
        if vdir.exists() and (vdir/"latent.npy").exists(): continue
        ensure_dir(vdir)
        t0=time.time()

        z_jazz=model.translate(z0,src_idx,tgt_idx,
                               prm["sigma"],prm["beta"],prm["seed"])
        np.save(vdir/"latent.npy",z_jazz.astype(np.float32))

        features_obj=None
        if args.decode_now and model.decoder_available:
            try:
                xh=model.decode_features_proxy(z_jazz,prm["temperature"])
                features_obj={n:float(v) for n,v in zip(feat_names,xh.tolist())}
                json.dump(features_obj,open(vdir/"features.json","w"),indent=2)
            except Exception as e:
                append_jsonl(batch_log,{"ts":utc_now(),"warn":str(e)})

        params={**prm,"src_genre":"classical","tgt_genre":"jazz",
                "created":utc_now()}
        json.dump(params,open(vdir/"params.json","w"),indent=2)

        cos=safe_cosine(z0,z_jazz)
        append_jsonl(batch_log,{"ts":utc_now(),"version":i,"cosine":cos,**prm})
        manifest["steps"]["classical_to_jazz"].append(
            {"version":i,"dir":vdir.name,"params":prm,
             "cosine_to_source":cos,"has_features":bool(features_obj)})
        json.dump(manifest,open(manifest_path,"w"),indent=2)
        print(f"v{i:02d} done in {time.time()-t0:.2f}s")

    print(f"\n✅ Done: {len(grid)} versions → {step_dir}\n")

# ------------------------------------------------------------------
if __name__=="__main__": main()