"""Paint the pale sky left between treetops with the night sky colour, after build_night.py.
usage: fix_treeline.py <name.webp> <name-depth.webp>  (rewrites the colour file in place)"""
import sys
import cv2
import numpy as np
from PIL import Image

COLOUR, DEPTH = sys.argv[1], sys.argv[2]
REACH, MIN_LUM = 121, 22  # how far from open sky to look, in px; anything brighter than the dark leaves is leftover sky
hero = np.asarray(Image.open(COLOUR).convert("RGB")).astype(np.float32)
d = np.asarray(Image.open(DEPTH).convert("L"))
sky = cv2.morphologyEx(((d < 10) & (hero.sum(2) > 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
near = cv2.dilate(sky, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REACH, REACH))) > 0
sky_colour = np.median(hero[sky > 0], axis=0)
lum = hero @ np.array([0.2126, 0.7152, 0.0722], np.float32)
grey = hero.max(2) - hero.min(2) < 0.45 * hero.max(2) + 1  # leftover sky is grey-blue, leaves are green
scene = (d > 10) & (d < 140)  # the figure sits at 0.6 depth and up, so he is never touched
weight = np.clip((lum - MIN_LUM) / 20, 0, 1) * (near & grey & scene)
weight = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), 1.2)[..., None]
out = np.clip(hero * (1 - weight) + sky_colour * weight + 0.5, 0, 255).astype(np.uint8)
Image.fromarray(out).save(COLOUR, quality=92, method=6)
