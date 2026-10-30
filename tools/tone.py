"""Colour helpers shared by the build scripts."""
import cv2
import numpy as np


def smoothstep(low, high, x):
    t = np.clip((x - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def to_srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def masked_blur(values, mask, sigma):
    """Blur that ignores pixels outside the mask, so silhouettes don't bleed into the high-pass."""
    weight = cv2.GaussianBlur(mask, (0, 0), sigma) + 1e-5
    return cv2.GaussianBlur(values * mask, (0, 0), sigma) / weight
