"""
CS 532 Homework 1 (c) - DLT using line feature locations (Corrected)
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Assuming you have a utils.py with this function or similar
def preprocess_image(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = np.array(Image.open(input_path).convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    Image.fromarray(enhanced_img).save(output_path)
    return enhanced_img

class LineHomographyDLT:
    def __init__(self, input_path):
        self.input_path = input_path
        self.img = preprocess_image(input_path, "images/output/enhanced_partC.jpg")

    def select_lines(self):
        names = ["Top", "Right", "Bottom", "Left"]
        lines, cur, idx = [], [], [0]

        def onclick(event):
            if event.inaxes and idx[0] < 4:
                cur.append([event.xdata, event.ydata])
                plt.plot(event.xdata, event.ydata, 'o', color='cyan' if len(cur) == 1 else 'red')
                if len(cur) == 2:
                    lines.append(cur.copy())
                    plt.plot([cur[0][0], cur[1][0]], [cur[0][1], cur[1][1]], 'y-')
                    cur.clear(); idx[0] += 1
                    plt.title(f"Click {names[idx[0]]} line's two endpoints" if idx[0] < 4 else "Done, close window")
                plt.draw()
        
        plt.figure(figsize=(10, 7))
        plt.imshow(self.img)
        plt.title(f"Click {names[0]} line's two endpoints")
        plt.connect('button_press_event', onclick)
        plt.show()

        if len(lines) < 4:
            raise ValueError("You must select 4 lines. Please run again.")

        return np.array(lines, dtype=np.float32)

    def line_eq(self, p1, p2):
        """Calculates the homogeneous representation of a line from two points."""
        x1, y1 = p1
        x2, y2 = p2
        # Using cross product [x1, y1, 1] x [x2, y2, 1]
        a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
        # Normalize for numerical stability
        norm = math.hypot(a, b)
        return np.array([a, b, c]) / norm if norm != 0 else np.array([a, b, c])

    def compute_line_homography(self, src_lines, dst_lines):
        """Calculates homography from line correspondences using DLT."""
        A = []
        for src_line_pts, dst_line_pts in zip(src_lines, dst_lines):
            # Get homogeneous line equations: l' ~ H^(-T)l or l ~ H^T l'
            # We will use the l ~ H^T l' formulation
            l_src = self.line_eq(src_line_pts[0], src_line_pts[1])
            l_dst = self.line_eq(dst_line_pts[0], dst_line_pts[1])
            
            # From the cross product l_src x (H^T l_dst) = 0, we get two
            # linearly independent equations for each line correspondence.
            # This forms two rows in the A matrix.
            row1 = [0, 0, 0,
                    -l_src[2] * l_dst[0], -l_src[2] * l_dst[1], -l_src[2] * l_dst[2],
                    l_src[1] * l_dst[0], l_src[1] * l_dst[1], l_src[1] * l_dst[2]]
            
            row2 = [l_src[2] * l_dst[0], l_src[2] * l_dst[1], l_src[2] * l_dst[2],
                    0, 0, 0,
                    -l_src[0] * l_dst[0], -l_src[0] * l_dst[1], -l_src[0] * l_dst[2]]

            A.append(row1)
            A.append(row2)
            
        _, _, Vt = np.linalg.svd(np.array(A))
        H_T = Vt[-1].reshape(3, 3)
        
        # We solved for H^T, so we need to transpose it back to get H
        H = H_T.T
        
        # Normalize the homography matrix
        return H / H[2, 2]

    def run(self):
        print("Please select the 4 boundary lines of the court by clicking their two endpoints for each line.")
        src_lines = self.select_lines()
        
        # Destination lines for the rectangular output
        dst_lines = np.array([
            [[0, 0], [939, 0]],          # Top line
            [[939, 0], [939, 499]],      # Right line
            [[939, 499], [0, 499]],      # Bottom line
            [[0, 499], [0, 0]]           # Left line
        ], dtype=np.float32)
        
        H = self.compute_line_homography(src_lines, dst_lines)
        
        print("\nCalculated Homography Matrix:\n", H)
        
        warped = cv2.warpPerspective(self.img, H, (940, 500))
        
        # Convert RGB to BGR for saving with OpenCV
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        cv2.imwrite("images/output/line_homography_result.jpg", warped_bgr)
        
        print("\nPart (c) complete — warped image saved as images/output/line_homography_result.jpg")

        # Display the result
        plt.figure(figsize=(10, 5))
        plt.imshow(warped)
        plt.title("Final Warped Image (from Lines)")
        plt.show()

if __name__ == "__main__":
    input_file = "images/input/basketball-court.ppm"
    if os.path.exists(input_file):
        LineHomographyDLT(input_file).run()
    else:
        print(f"Error: Input file not found at '{input_file}'")