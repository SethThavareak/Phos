"""Stage 2: turn the daylight photo into the night plate + depth map the relight effect needs.

Runs locally. usage: build_hero.py <photo> <maps dir from masks.py> <public/textures>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from tone import masked_blur, smoothstep, to_linear, to_srgb

SRC, MAPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SKY, PERSON = 2, 12  # ADE20K ids

CANVAS_W, CANVAS_H = 2400, 1350
CROP_TOP, CROP_BOTTOM = 120, 1240  # trim empty sky and the waist-down
MIN_SURFACE_DEPTH, SKY_DEPTH = 0.12, 0.06  # keep SKY_DEPTH in sync with plane.js
MAX_SKY_LUM = 0.03  # brightest the night sky may be before the glow turns it white
LEFT_FADE = 0.4  # share of the photo width that melts into the dark copy area


class Depth:
    def __init__(self):
        name = "depth-anything/Depth-Anything-V2-Large-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()

    def crop(self, image, box):
        left, top, right, bottom = box
        crop = Image.fromarray(image[top:bottom, left:right])
        with torch.no_grad():
            inputs = self.processor(images=crop, return_tensors="pt", size={"height": 1022, "width": 1022}, do_resize=True, keep_aspect_ratio=True, ensure_multiple_of=14).to(self.device)
            out = self.model(**inputs).predicted_depth[None]
            out = torch.nn.functional.interpolate(out, size=(bottom - top, right - left), mode="bicubic", align_corners=False)
        out = out[0, 0].cpu().numpy()
        out = (out - out.min()) / (out.max() - out.min())
        full = np.zeros(image.shape[:2], np.float32)
        full[top:bottom, left:right] = out
        return full


def detail(crop_depth, mask, sigma):
    return crop_depth - masked_blur(crop_depth, mask, sigma)


def window(shape, box, feather):
    """Soft-edged box, so a crop's detail doesn't end in a visible seam."""
    mask = np.zeros(shape, np.float32)
    left, top, right, bottom = box
    mask[top + feather : bottom - feather, left + feather : right - feather] = 1
    return cv2.GaussianBlur(mask, (0, 0), feather / 2)


rgb = np.asarray(Image.open(SRC).convert("RGB"))
depth = np.load(MAPS / "depth.npy")
labels = np.load(MAPS / "labels.npy")

# Masks ---------------------------------------------------------------------
person = ((labels == PERSON) | (depth > 0.74)).astype(np.uint8)
_, components = cv2.connectedComponents(person)
sizes = np.bincount(components.ravel())[1:]
person = (components == 1 + sizes.argmax()).astype(np.float32)
person = cv2.GaussianBlur(person, (0, 0), 1.2)

srgb = rgb.astype(np.float32) / 255
lum = srgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
near_sky = cv2.dilate(((labels == SKY) | (depth < 0.05)).astype(np.uint8), np.ones((9, 9), np.uint8))
sky = smoothstep(0.24, 0.4, lum) * near_sky * (1 - person)  # full-res luminance matte keeps the palm fronds
sky[depth > 0.2] = 0
# Depth undershoots beside the hair, so bright stone there passes as sky; the figure never touches real sky.
near_figure = cv2.dilate((person > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
sky[near_figure > 0] = 0
sky = cv2.GaussianBlur(sky, (0, 0), 0.8)  # anti-alias the matte, it comes from a low-quality JPEG

# Depth ---------------------------------------------------------------------
# Monocular depth spends its whole range on the subject; re-run on crops to get relief out of the
# temple and the face, then inject that detail into a global depth that gives the far plane room.
estimator = Depth()
far = (1 - person) * (1 - sky) * (depth < 0.45)
near = person
temple_box, person_box, face_box = (0, 170, 960, 800), (400, 520, 960, 1280), (560, 530, 800, 810)
temple_detail = detail(estimator.crop(rgb, temple_box), far, 30) * window(depth.shape, temple_box, 60)
person_detail = detail(estimator.crop(rgb, person_box), near, 45) * window(depth.shape, person_box, 20)
face_detail = detail(estimator.crop(rgb, face_box), near, 16) * window(depth.shape, face_box, 30)

# Close gaps between fence slats and fronds: each one is a pit that fills with shadow.
closed = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
background = (1 - person) * (1 - sky)
closed = cv2.medianBlur((closed * 255).astype(np.uint8), 9).astype(np.float32) / 255
depth = depth * (1 - background) + closed * background

# Compressed on purpose: a big subject-to-backdrop jump renders as a black outline.
shaped = 0.75 * depth ** 0.6
# Eroded and clamped: detail spikes on the silhouette can otherwise push the figure's edge down to sky depth.
core = cv2.erode(near, np.ones((9, 9), np.uint8))
shaped += temple_detail.clip(-0.1, 0.1) * far * 0.4
shaped += (person_detail.clip(-0.1, 0.1) * 0.24 + face_detail.clip(-0.1, 0.1) * 0.08) * core
# The shader makes depth below SKY_DEPTH glow, so only real sky may sit under it.
shaped = np.clip(np.maximum(shaped, MIN_SURFACE_DEPTH) * (1 - sky), 0, 1)

# Plate ---------------------------------------------------------------------
smooth = cv2.fastNlMeansDenoisingColored(rgb, None, 2, 4, 5, 15).astype(np.float32) / 255
linear = to_linear(smooth)
lifted = linear ** 0.72  # recover the backlit stone so the light has something to reveal
far_soft = cv2.GaussianBlur(far, (0, 0), 3)[..., None]
linear = linear * (1 - far_soft) + lifted * far_soft
linear = linear * (1 - person[..., None]) * 0.8 + linear * person[..., None]

clouds = smoothstep(0.42, 0.82, lum)[..., None]
night_top = np.array([0.0105, 0.0135, 0.0245], np.float32)
night = np.array([0.0026, 0.0040, 0.0105], np.float32) * (1 - clouds) + night_top * clouds
# Squared so sky-coloured fringe pixels on silhouettes don't survive into the glow.
plate = linear * (1 - sky[..., None]) ** 2 + night * sky[..., None]

# Composition ---------------------------------------------------------------
plate = plate[CROP_TOP:CROP_BOTTOM]
shaped = shaped[CROP_TOP:CROP_BOTTOM]
scale = CANVAS_H / plate.shape[0]
width = round(plate.shape[1] * scale)
plate = cv2.resize(plate, (width, CANVAS_H), interpolation=cv2.INTER_LANCZOS4)
shaped = cv2.resize(shaped, (width, CANVAS_H), interpolation=cv2.INTER_CUBIC)

fade = smoothstep(0, LEFT_FADE, np.linspace(0, 1, width, dtype=np.float32))[None, :]
canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.float32)
canvas_depth = np.zeros((CANVAS_H, CANVAS_W), np.float32)
canvas[:, CANVAS_W - width :] = plate * fade[..., None]
canvas_depth[:, CANVAS_W - width :] = shaped  # only the colour fades: a depth ramp would cast a shadow wall

# Resizing nudges colour and depth apart by a pixel; whatever lands in the glow band must be no brighter than sky.
band = canvas_depth < SKY_DEPTH
canvas[band] = np.minimum(canvas[band], night_top)

OUT.mkdir(parents=True, exist_ok=True)
hero = (to_srgb(canvas) * 255 + 0.5).astype(np.uint8)
Image.fromarray(hero).save(OUT / "hero.webp", quality=88, method=6)
Image.fromarray((np.clip(canvas_depth, 0, 1) * 255 + 0.5).astype(np.uint8)).save(OUT / "hero-depth.webp", quality=95, method=6)

figure_canvas = cv2.resize(near_figure[CROP_TOP:CROP_BOTTOM], (width, CANVAS_H), interpolation=cv2.INTER_NEAREST)
assert not (canvas_depth[:, CANVAS_W - width :] < SKY_DEPTH)[figure_canvas > 0].any(), "figure edge fell into the sky glow band"

# Checked on the decoded files, since that's what the shader sees after lossy compression.
saved = to_linear(np.asarray(Image.open(OUT / "hero.webp").convert("RGB"), np.float32) / 255)
saved_depth = np.asarray(Image.open(OUT / "hero-depth.webp").convert("L"), np.float32) / 255
# Same falloff as the shader's depth.smoothstep(SKY_DEPTH, 0).
glowing = (saved @ np.array([0.2126, 0.7152, 0.0722], np.float32)) * (1 - smoothstep(0, SKY_DEPTH, saved_depth))
assert glowing.max() <= MAX_SKY_LUM, f"non-sky pixels would glow: {int((glowing > MAX_SKY_LUM).sum())} over limit, max {glowing.max():.3f}"
assert hero.shape == (CANVAS_H, CANVAS_W, 3) and hero[:, :40].max() == 0, "copy area must stay black"
for name in ("hero.webp", "hero-depth.webp"):
    print(name, (OUT / name).stat().st_size // 1024, "KB")
