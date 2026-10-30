"""Stage 1.5: divide hard sunlight out of a photo, so the page's moving light is the only light on it.

Estimates how much light reached the shirt, skin and scene, then evens each out. Fabric folds and print detail
stay, because they are finer than the estimate. Tone only: nothing is generated.
Runs locally. usage: even_light.py <photo> <maps dir from masks.py> <output photo>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tone import masked_blur, to_linear, to_srgb

SRC, MAPS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PERSON = 12  # ADE20K id
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SHIRT_STRENGTH, SKIN_STRENGTH, SCENE_STRENGTH = 0.85, 0.55, 0.6  # below 1 so surfaces keep some form
NECK = 0.5  # share of the photo height
SKIN_BLUR, SCENE_BLUR = 14, 70  # wide enough to skip features and print, narrow enough to follow the light
MAX_GAIN = 4


def edge_aware(values, mask):
    """Follows the hard shadow edge but not the folds: the mask is filled first so its surroundings don't bleed in."""
    filled = values * mask + masked_blur(values, mask, 15) * (1 - mask)
    return cv2.bilateralFilter(filled, -1, 0.6, 16)  # full size: a smaller estimate leaves a ridge along the edge


def even(linear, light, mask, strength, feather=1.5):
    target = np.median(light[mask > 0.5])
    gain = np.clip(np.exp((target - light) * strength), 1 / MAX_GAIN, MAX_GAIN)
    return linear * (1 + (gain[..., None] - 1) * cv2.GaussianBlur(mask, (0, 0), feather)[..., None])


srgb = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255
labels = np.load(MAPS / "labels.npy")
linear = to_linear(srgb)
luma = linear @ LUMA
log_luma = np.log(np.maximum(luma, 1e-3))
saturation = srgb.max(2) - srgb.min(2)

person = (labels == PERSON).astype(np.uint8)
# Dark enough to take in the shaded fabric, which is the point; the head is cut off because dark hair would pass too.
# Judged on the neighbourhood too: sunlit knuckles wash out to a shirt's colour, the skin around them doesn't.
warmth = srgb[..., 0] - srgb[..., 2]
shirt = ((person > 0) & (saturation < 0.15) & (cv2.GaussianBlur(saturation, (0, 0), 10) < 0.15) & (warmth < 0.09) & (luma > 0.025)).astype(np.uint8)
shirt[: round(shirt.shape[0] * NECK)] = 0
shirt = cv2.morphologyEx(shirt, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(np.float32)
# Saturated only: shine on the nose stays out, because darkening a white highlight turns it grey.
skin = ((person > 0) & (shirt == 0) & (saturation > 0.15) & (warmth > 0.08)).astype(np.float32)
scene = 1 - cv2.dilate(person, np.ones((9, 9), np.uint8)).astype(np.float32)

evened = even(linear, edge_aware(log_luma, shirt), shirt, SHIRT_STRENGTH)
evened = even(evened, masked_blur(log_luma, skin, SKIN_BLUR), skin, SKIN_STRENGTH, feather=6)  # soft, or the mask's edge shows on skin
evened = even(evened, masked_blur(log_luma, scene, SCENE_BLUR), scene, SCENE_STRENGTH)

result = (to_srgb(evened) * 255 + 0.5).astype(np.uint8)
Image.fromarray(result).save(OUT, quality=97, subsampling=0)
print(OUT, f"shirt {shirt.mean():.2f} skin {skin.mean():.2f} scene {scene.mean():.2f}")
