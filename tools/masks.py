"""Stage 1: depth and semantic masks for the source photo (local GPU, no uploads)."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
)

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
image = Image.open(SRC).convert("RGB")
size = image.size[::-1]

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
with torch.no_grad():
    # shortcut: fixed 1022px inference, tile it if the source ever gets much larger
    inputs = processor(images=image, return_tensors="pt", size={"height": 1022, "width": 1022}, do_resize=True, keep_aspect_ratio=True, ensure_multiple_of=14).to(device)
    depth = model(**inputs).predicted_depth[None]
    depth = torch.nn.functional.interpolate(depth, size=size, mode="bicubic", align_corners=False)
depth = depth[0, 0].cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min())
np.save(OUT / "depth.npy", depth.astype(np.float32))

SEG_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
with torch.no_grad():
    logits = model(**processor(images=image, return_tensors="pt").to(device)).logits
    logits = torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
labels = logits[0].argmax(0).cpu().numpy()
np.save(OUT / "labels.npy", labels.astype(np.uint8))

names = model.config.id2label
present = {names[int(i)]: round(float((labels == i).mean()), 3) for i in np.unique(labels)}
print(dict(sorted(present.items(), key=lambda kv: -kv[1])))
Image.fromarray((depth * 255).astype(np.uint8)).save(OUT / "depth-preview.png")
