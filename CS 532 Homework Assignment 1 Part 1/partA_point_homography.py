import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class HomographyDLTPoints:
    def __init__(self, input_path):
        self.input_path = input_path
        self.enhanced_img = None
        # Create the output directory if it doesn't exist
        os.makedirs("images/output", exist_ok=True)

    def preprocess(self):
        """Enhances the image contrast for easier point selection."""
        img = np.array(Image.open(self.input_path).convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        self.enhanced_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        Image.fromarray(self.enhanced_img).save("images/output/enhanced.jpg")
        print("Enhanced image saved to images/output/enhanced.jpg")

    def select_points(self):
        """Displays the image and records four user clicks for the source points."""
        labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        coords = []

        def onclick(event):
            if len(coords) < 4 and event.inaxes:
                x, y = event.xdata, event.ydata
                coords.append([x, y])
                plt.plot(x, y, "ro")
                plt.text(x + 5, y - 5, f"P{len(coords)}", color="yellow")
                plt.title(
                    f"Click {labels[len(coords)]}" if len(coords) < 4 else "Done — close window to continue"
                )
                plt.draw()

        plt.figure(figsize=(10, 7))
        plt.imshow(self.enhanced_img)
        plt.title(f"Click {labels[0]}")
        plt.connect("button_press_event", onclick)
        plt.show()

        if len(coords) < 4:
            raise ValueError("You must select 4 points. Please run again.")
            
        return np.array(coords, np.float32)

    def compute_homography(self, src, dst):
        """Calculates the homography matrix using the DLT algorithm."""
        A = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        
        # Use SVD to find the solution for Ah = 0
        _, _, Vt = np.linalg.svd(np.array(A))
        
        # The homography matrix is the last row of Vt
        H = Vt[-1].reshape(3, 3)
        
        # Normalize the matrix
        return H / H[2, 2]

    def warp_image(self, H, output_size):
        """Warps the enhanced image using the calculated homography matrix."""
        print("\nPerforming image warping...")
        # cv2.warpPerspective requires (width, height) for the output size
        warped = cv2.warpPerspective(self.enhanced_img, H, output_size)
        
        # Save the final warped image
        Image.fromarray(warped).save("images/output/PartA_court.jpg")

        # Display the result in a new window
        plt.figure(figsize=(10, 5))
        plt.imshow(warped)
        plt.title("Final Warped Image")
        plt.show()

    def run(self):
        """Executes the entire pipeline from preprocessing to warping."""
        self.preprocess()
        
        print("Please click the 4 corners of the basketball court.")
        src_pts = self.select_points()
        
        # Destination points define the rectangular shape of the output image
        dst_pts = np.array([[0, 0], [939, 0], [939, 499], [0, 499]], np.float32)

        # Calculate the homography
        H = self.compute_homography(src_pts, dst_pts)
        np.save("images/output/H_partA.npy", H)
        print("\nHomography matrix saved as images/output/H_partA.npy")
        print("Matrix:\n", H)

        # Apply the homography to warp the image
        output_dimensions = (940, 500)  # (width, height)
        self.warp_image(H, output_dimensions)

if __name__ == "__main__":
    # Make sure you have an 'images/input' directory with this file
    # or change the path to your image file.
    input_file = "images/input/basketball-court.ppm"
    if os.path.exists(input_file):
        HomographyDLTPoints(input_file).run()
    else:
        print(f"Error: Input file not found at '{input_file}'")
        print("Please create an 'images/input' folder and place the image there.")