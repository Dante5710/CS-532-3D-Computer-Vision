# ---------------- Part (b): Bilinear Interpolation ----------------
import cv2
import numpy as np
from PIL import Image
import os

class BilinearWarp:
    def __init__(self, input_path):
        self.input_path = input_path
        self.enhanced_img = None
        os.makedirs("images/output", exist_ok=True)

    def preprocess(self):
        img = np.array(Image.open(self.input_path).convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
        self.enhanced_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    def bilinear(self, img, x, y):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)
        dx, dy = x - x0, y - y0
        tl, tr, bl, br = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
        return ((1 - dy) * ((1 - dx) * tl + dx * tr) + dy * ((1 - dx) * bl + dx * br)).astype(np.uint8)

    def warp(self, H, size=(940, 500)):
        Hinv = np.linalg.inv(H)
        out = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for y in range(size[1]):
            for x in range(size[0]):
                X = Hinv @ np.array([x, y, 1])
                if X[2] != 0:
                    sx, sy = X[0] / X[2], X[1] / X[2]
                    if 0 <= sx < self.enhanced_img.shape[1] and 0 <= sy < self.enhanced_img.shape[0]:
                        out[y, x] = self.bilinear(self.enhanced_img, sx, sy)
        return out

    def run(self):
        self.preprocess()

        #  Load homography from Part (a)
        H = np.load("images/output/H_partA.npy")
        print("Loaded Homography:\n", H)

        warped = self.warp(H)
        cv2.imwrite("images/output/partB_bilinear_result.jpg", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        print(" Warped image saved as images/output/partB_bilinear_result.jpg")

if __name__ == "__main__":
    BilinearWarp("images/input/basketball-court.ppm").run()
