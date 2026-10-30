"""Stage 2 for a plain studio photo: no sky to replace, just a dark wall for the figure to stand off.

Runs locally. usage: build_studio.py <photo> <maps dir from masks.py> <src/textures>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tone import masked_blur, smoothstep, to_linear, to_srgb

SRC, MAPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PERSON = 12  # ADE20K id

CANVAS_W, CANVAS_H = 3200, 1800
# Scaled to fit, never cropped: the whole body shows, with the head as low as hero.webp's and room above it for phone copy.
PHOTO_LEFT, PHOTO_TOP = 1974, 320
WALL = np.array([0.075, 0.080, 0.092], np.float32)  # dark enough that the light has room to climb
# Flat on purpose: any relief on the figure turned away from side light as dark, hard-edged blotches on the skin.
WALL_DEPTH, FIGURE_DEPTH = 0.4, 0.64  # keep WALL_DEPTH in sync with app.js; plane.js exempts the figure from 0.6 up
NECK = 0.47  # share of the photo height
FADE_START, FADE_END = 1067, 2333  # canvas x: wide, or the pool of light ends in a visible edge at the copy area


def resize(image, interpolation):
    return cv2.resize(image, (round(image.shape[1] * scale), CANVAS_H - PHOTO_TOP), interpolation=interpolation)


rgb = np.asarray(Image.open(SRC).convert("RGB"))
labels = np.load(MAPS / "labels.npy")
scale = (CANVAS_H - PHOTO_TOP) / rgb.shape[0]

srgb = resize(rgb.astype(np.float32) / 255, cv2.INTER_AREA)
linear = to_linear(srgb)
seg = resize((labels == PERSON).astype(np.uint8), cv2.INTER_NEAREST)

# Masks ---------------------------------------------------------------------
# Segmentation is coarse at this size, but against a plain wall colour difference gives a crisp edge.
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
sure_wall = (1 - cv2.dilate(seg, ellipse)).astype(np.float32)
# A wide blur: narrower ones go dark mid-torso, and the dark shirt then passes for wall.
wall_colour = masked_blur(srgb, sure_wall[..., None].repeat(3, 2), 150)
difference = np.abs(srgb - wall_colour).max(2)
matte = smoothstep(0.1, 0.2, difference) * cv2.dilate(seg, ellipse)
# Segmentation only fills pale skin back in: it also claims the wall showing between the arms and the torso.
core = cv2.erode(seg, ellipse)
matte = np.maximum(matte, core * smoothstep(0.03, 0.08, difference))
# Shaded wall in those gaps is too far from the wall colour, so catch it as bright and neutral; the face keeps its eye whites.
core[round(core.shape[0] * NECK) :] = 0
wallish = smoothstep(0.35, 0.45, srgb.mean(2)) * (1 - smoothstep(0.05, 0.1, srgb.max(2) - srgb.min(2)))
matte *= 1 - wallish * (1 - core)
_, components = cv2.connectedComponents((matte > 0.5).astype(np.uint8))
sizes = np.bincount(components.ravel())[1:]
person = (components == 1 + sizes.argmax()).astype(np.uint8)
person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# The matte's threshold still keeps pixels that are mostly wall, which show as a pale line on the dark wall.
person = cv2.erode(person, np.ones((3, 3), np.uint8))
alpha = cv2.GaussianBlur(person.astype(np.float32), (0, 0), 1.2)

# Plate ---------------------------------------------------------------------
# Only the thin fringe still holding wall colour is refilled: a wider band reads as smeared skin around the outline.
inside = cv2.erode(person, np.ones((5, 5), np.uint8)).astype(np.float32)
inside_soft = cv2.GaussianBlur(inside, (0, 0), 1)[..., None]
figure = linear * inside_soft + masked_blur(linear, inside[..., None].repeat(3, 2), 2) * (1 - inside_soft)

# Depth ---------------------------------------------------------------------
# Matches the colour edge exactly: any figure pixel left at wall depth catches the wall's shadow.
shaped = WALL_DEPTH * (1 - alpha) + FIGURE_DEPTH * alpha

# Composition ---------------------------------------------------------------
photo_w = alpha.shape[1]
fade = smoothstep(FADE_START, FADE_END, np.arange(CANVAS_W, dtype=np.float32))[None, :, None]
canvas = np.ones((CANVAS_H, CANVAS_W, 3), np.float32) * WALL * fade
canvas_depth = np.full((CANVAS_H, CANVAS_W), WALL_DEPTH, np.float32)
region = (slice(PHOTO_TOP, CANVAS_H), slice(PHOTO_LEFT, PHOTO_LEFT + photo_w))
canvas[region] = canvas[region] * (1 - alpha[..., None]) + figure * alpha[..., None]
canvas_depth[region] = shaped

OUT.mkdir(parents=True, exist_ok=True)
# Dithered: a fade this slow spans few 8-bit levels, which show as vertical stripes once the light multiplies them.
dither = np.random.default_rng(0).uniform(-1, 1, canvas.shape[:2])[..., None] * (canvas > 0).any(2, keepdims=True)
hero = np.clip(to_srgb(canvas) * 255 + 0.5 + dither, 0, 255).astype(np.uint8)
# High quality and lossless: the normals pick up block edges from either file as a plaid pattern on the shirt.
Image.fromarray(hero).save(OUT / "sokheng.webp", quality=97, method=6)
Image.fromarray((np.clip(canvas_depth, 0, 1) * 255 + 0.5).astype(np.uint8)).save(OUT / "sokheng-depth.webp", lossless=True, method=6)

# Checked on the decoded files, since that's what the shader sees after lossy compression.
saved_depth = np.asarray(Image.open(OUT / "sokheng-depth.webp").convert("L"), np.float32) / 255
assert saved_depth.min() >= 0.12, "only sky may fall into the plane.js glow band"
assert hero.shape == (CANVAS_H, CANVAS_W, 3) and hero[:, :40].max() == 0, "copy area must stay black"
figure_core = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
figure_core[region] = cv2.erode(person, np.ones((9, 9), np.uint8))
figure_floor = saved_depth[figure_core > 0].min()
assert figure_floor >= 0.6, f"figure would pick up guessed self-shadow: depth {figure_floor:.3f}"
for name in ("sokheng.webp", "sokheng-depth.webp"):
    print(name, (OUT / name).stat().st_size // 1024, "KB")
