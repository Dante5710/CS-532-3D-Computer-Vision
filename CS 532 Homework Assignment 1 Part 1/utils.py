"""
utils.py - Shared utilities
Contains:
- CLAHE-based preprocessing
- Bilinear interpolation
"""

import cv2, os
import numpy as np
from PIL import Image

def preprocess_image(input_path, output_path=None):
    """Enhance image contrast using CLAHE."""
    os.makedirs('images/output', exist_ok=True)
    img = np.array(Image.open(input_path).convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    if output_path:
        Image.fromarray(enhanced).save(output_path)
    return enhanced

def bilinear_interpolation(img, x, y):
    """Manually perform 2D bilinear interpolation."""
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
    dx, dy = x - x0, y - y0
    tl, tr = img[y0, x0], img[y0, x1]
    bl, br = img[y1, x0], img[y1, x1]
    return ((1 - dy) * ((1 - dx) * tl + dx * tr) + dy * ((1 - dx) * bl + dx * br)).astype(np.uint8)
