"""Stage 2 for a photo with a real scene: the same night treatment as build_hero.py, on the whole uncropped frame.

The canvas height is the photo height plus top padding, which lowers a head that sits too high for the phone copy.
Runs locally. usage: build_night.py <photo> <maps dir from masks.py> <src/textures> <name> [top padding px] [segmentation inset px] [wall depth]
A wall depth flattens the scene onto one plane, for rooms where a plain wall would show the outline of any depth edge.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tone import masked_blur, smoothstep, to_linear, to_srgb

SRC, MAPS, OUT, NAME = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
TOP_PAD = int(sys.argv[5]) if len(sys.argv) > 5 else 0
SEG_INSET = int(sys.argv[6]) if len(sys.argv) > 6 else 0
WALL_DEPTH = float(sys.argv[7]) if len(sys.argv) > 7 else 0  # keep in sync with wallDepth in app.js
SKY, PERSON = 2, 12  # ADE20K ids

MIN_SURFACE_DEPTH, SKY_DEPTH = 0.12, 0.06  # keep SKY_DEPTH in sync with plane.js
MAX_SKY_LUM = 0.03  # brightest the night sky may be before the glow turns it white
LEFT_FADE = 0.4  # share of the photo width that melts into the dark copy area
TOP_FADE = 0.3  # share of the photo height that melts into the top padding
NIGHT_SKY = np.array([0.0052, 0.0074, 0.0165], np.float32)
SCENE_GAIN, SCENE_DESATURATE = 0.42, 0.4  # sunlit leaves are far brighter than anything at night, and go neon under warm light
GATE_DEPTH, GATE_GAIN = 0.7, 0.75  # the wooden gate frames the shot, so it stays close to its real colour
FIGURE_DEPTH_MIN, FIGURE_RELIEF, FIGURE_ROUNDING = 0.62, 0.1, 0.08  # plane.js exempts the figure from 0.6 up


def ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def largest(mask):
    _, components = cv2.connectedComponents(mask.astype(np.uint8))
    return (components == 1 + np.bincount(components.ravel())[1:].argmax()).astype(np.uint8)


rgb = np.asarray(Image.open(SRC).convert("RGB"))
depth = np.load(MAPS / "depth.npy")
labels = np.load(MAPS / "labels.npy")
height, width = rgb.shape[:2]
CANVAS_H = height + TOP_PAD
CANVAS_W = round(CANVAS_H * 16 / 9)

srgb = rgb.astype(np.float32) / 255
lum = srgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)

# Masks ---------------------------------------------------------------------
# Segmentation finds him but runs loose; the depth edge is tighter, so split each row halfway between him and what's behind him.
seg = largest(labels == PERSON)
near = cv2.dilate(seg, ellipse(25))
ring = cv2.dilate(seg, ellipse(61)) - cv2.dilate(seg, ellipse(21))
split = np.array([(np.median(depth[y][seg[y] > 0]) + np.median(depth[y][ring[y] > 0])) / 2 if seg[y].any() else 1 for y in range(height)], np.float32)
split = cv2.GaussianBlur(split[:, None], (0, 0), 25)
person = (depth > split) & (near > 0)
if SEG_INSET:  # keeps what the depth edge can't separate, like a bare foot on the floor
    person |= cv2.erode(seg, ellipse(SEG_INSET)) > 0
person = largest(person)
person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, ellipse(9))
if WALL_DEPTH:
    # The gaps between limbs let pale ground through; the top half is spared for eye whites and whatever he's holding.
    pale = smoothstep(0.45, 0.55, lum) * (1 - smoothstep(0.05, 0.1, srgb.max(2) - srgb.min(2))) > 0.3
    pale[: height // 2] = False
    pale = cv2.dilate(pale.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # the reach catches a gap's thin ends
    # A white shirt is pale too; clearing it would eat grey bands out of the shoulders.
    if pale[person > 0].mean() < 0.15:
        person[pale & (person - cv2.erode(person, ellipse(25)) > 0)] = 0
    person = largest(person)
    person = cv2.erode(person, np.ones((3, 3), np.uint8))  # pixels that are mostly wall show as a pale line
alpha = cv2.GaussianBlur(person.astype(np.float32), (0, 0), 1.2)
inside = cv2.erode(person, ellipse(9)).astype(np.float32)

# Only pale pixels in the sky's neighbourhood count, which keeps the leaves crossing it.
near_sky = cv2.dilate((labels == SKY).astype(np.uint8), ellipse(25))
sky = smoothstep(0.55, 0.75, lum) * near_sky * (depth < 0.2)
sky = cv2.GaussianBlur(sky, (0, 0), 1)

# Depth ---------------------------------------------------------------------
# Close gaps between leaves: each one is a pit that fills with shadow.
scene = (1 - alpha) * (1 - sky)
closed = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, ellipse(21))
closed = cv2.medianBlur((closed * 255).astype(np.uint8), 9).astype(np.float32) / 255
depth_scene = depth * (1 - scene) + closed * scene
shaped = 0.75 * depth_scene ** 0.6

low, high = np.percentile(depth[inside > 0], [2, 98])
relief = masked_blur(np.clip((depth - low) / (high - low), 0, 1), person.astype(np.float32), 16)
rounding = np.sqrt(np.clip(cv2.distanceTransform(person, cv2.DIST_L2, 5) / 160, 0, 1))
rounding = masked_blur(rounding, person.astype(np.float32), 34)  # blurred hard: a distance map creases along the body's centre line
figure = FIGURE_DEPTH_MIN + FIGURE_RELIEF * relief + FIGURE_ROUNDING * rounding
shaped = shaped * (1 - alpha) + figure * alpha
# The shader makes depth below SKY_DEPTH glow, so only real sky may sit under it.
shaped = np.clip(np.maximum(shaped, MIN_SURFACE_DEPTH) * (1 - sky), 0, 1)
if WALL_DEPTH:
    shaped = (WALL_DEPTH * (1 - alpha) + FIGURE_DEPTH_MIN * alpha) * (1 - sky)

# Plate ---------------------------------------------------------------------
linear = to_linear(srgb)
grey = (linear @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
night = (linear + (grey - linear) * SCENE_DESATURATE) * SCENE_GAIN
gate = smoothstep(GATE_DEPTH, GATE_DEPTH + 0.06, depth)[..., None]
night = night * (1 - gate) + linear * GATE_GAIN * gate
lit = linear
if WALL_DEPTH:  # only the thin fringe holding wall colour: a wider refill smears the outline
    fringe = cv2.erode(person, np.ones((5, 5), np.uint8)).astype(np.float32)
    fringe_soft = cv2.GaussianBlur(fringe, (0, 0), 1)[..., None]
    lit = linear * fringe_soft + masked_blur(linear, fringe[..., None].repeat(3, 2), 2) * (1 - fringe_soft)
plate = lit * alpha[..., None] + night * (1 - alpha[..., None])
# Squared so sky-coloured fringe pixels on the leaves don't survive into the glow.
plate = plate * (1 - sky[..., None]) ** 2 + NIGHT_SKY * sky[..., None]

# Composition ---------------------------------------------------------------
fade = smoothstep(0, LEFT_FADE, np.linspace(0, 1, width, dtype=np.float32))[None, :, None]
if TOP_PAD:
    fade = fade * smoothstep(0, TOP_FADE, np.linspace(0, 1, height, dtype=np.float32))[:, None, None]
fade = fade * (1 - alpha[..., None]) + alpha[..., None]  # only the scene fades: he stays whole
canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.float32)
canvas_depth = np.zeros((CANVAS_H, CANVAS_W), np.float32)
canvas[TOP_PAD:, CANVAS_W - width :] = plate * fade
canvas_depth[TOP_PAD:, CANVAS_W - width :] = shaped  # only the colour fades: a depth ramp would cast a shadow wall

# Whatever lands in the glow band must be no brighter than sky.
band = canvas_depth < SKY_DEPTH
canvas[band] = np.minimum(canvas[band], NIGHT_SKY)

OUT.mkdir(parents=True, exist_ok=True)
hero = to_srgb(canvas) * 255 + 0.5
if sky.mean() > 0.2:  # a big dark sky fading out spans few 8-bit levels, which show as stripes once the glow doubles them
    hero += np.random.default_rng(0).uniform(-1, 1, canvas.shape[:2])[..., None] * (canvas > 0).any(2, keepdims=True)
hero = np.clip(hero, 0, 255).astype(np.uint8)
Image.fromarray(hero).save(OUT / f"{NAME}.webp", quality=92, method=6)
Image.fromarray((np.clip(canvas_depth, 0, 1) * 255 + 0.5).astype(np.uint8)).save(OUT / f"{NAME}-depth.webp", quality=95, method=6)

# Checked on the decoded files, since that's what the shader sees after lossy compression.
saved = to_linear(np.asarray(Image.open(OUT / f"{NAME}.webp").convert("RGB"), np.float32) / 255)
saved_depth = np.asarray(Image.open(OUT / f"{NAME}-depth.webp").convert("L"), np.float32) / 255
# Same falloff as the shader's depth.smoothstep(SKY_DEPTH, 0).
glowing = (saved @ np.array([0.2126, 0.7152, 0.0722], np.float32)) * (1 - smoothstep(0, SKY_DEPTH, saved_depth))
assert glowing.max() <= MAX_SKY_LUM, f"non-sky pixels would glow: {int((glowing > MAX_SKY_LUM).sum())} over limit, max {glowing.max():.3f}"
assert hero.shape == (CANVAS_H, CANVAS_W, 3) and hero[:, :40].max() == 0, "copy area must stay black"
core = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
core[TOP_PAD:, CANVAS_W - width :] = cv2.erode(person, ellipse(25))
assert saved_depth[core > 0].min() >= 0.6, "figure would pick up guessed self-shadow"
assert person.sum() > 0.1 * person.size, "lost the figure"
for name in (f"{NAME}.webp", f"{NAME}-depth.webp"):
    print(name, (OUT / name).stat().st_size // 1024, "KB")
Image.fromarray(hero[:, CANVAS_W - width :]).resize((675, 675 * CANVAS_H // width), Image.LANCZOS).save(MAPS / "plate-preview.jpg", quality=90)
