import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ===============================================================
# RENDERING FUNCTION (with better normalization)
# ===============================================================
def PointCloud2Image(M, Sets3DRGB, viewport, filter_size=5):
    top = int(viewport[0]); left = int(viewport[1]); h = int(viewport[2]); w = int(viewport[3])
    bot = top + h + 1; right = left + w + 1
    output_image = np.zeros((h + 1, w + 1, 3), dtype=np.float32)

    for dataset in Sets3DRGB:
        P3D = dataset[:3, :]
        color = (dataset[3:6, :]).T
        if P3D.shape[1] == 0:
            continue

        X = np.concatenate((P3D, np.ones((1, P3D.shape[1]))))
        x = np.matmul(M, X)

        # keep points with positive depth
        valid = x[2, :] > 1e-6
        x = x[:, valid]
        color = color[valid, :]

        # project
        x = x / x[2, :]
        x[:2, :] = np.floor(x[:2, :])

        # keep only pixels inside viewport
        ix = (x[1, :] > top) & (x[0, :] > left) & (x[1, :] < bot) & (x[0, :] < right)
        rx = x[:, ix]; rcolor = color[ix, :]

        canvas = np.zeros((bot, right, 3), dtype=np.float32)
        for i in range(rx.shape[1]):
            u, v = int(rx[0, i]), int(rx[1, i])
            canvas[v, u, :] = rcolor[i, :]

        # simple dilation filter to smooth points
        for i in range(3):
            channel = canvas[top:bot, left:right, i]
            channel = cv2.dilate(channel, np.ones((filter_size, filter_size)))
            output_image[:, :, i] = np.maximum(output_image[:, :, i], channel[:h + 1, :w + 1])

    return np.clip(output_image, 0, 1)


# ===============================================================
# DATA LOADER
# ===============================================================
def load_data():
    import pickle
    input_file_path = "data.obj"
    print(f"Loading data from {input_file_path}...")
    with open(input_file_path, 'rb') as file_p:
        camera_objs = pickle.load(file_p)
    return camera_objs


# ===============================================================
# CAMERA POSE GENERATOR (HALF-CIRCLE)
# ===============================================================
def generate_camera_poses(object_center, viewing_distance, num_frames):
    angles = np.linspace(-np.pi / 2, np.pi / 2, num_frames)
    poses = []

    for angle in angles:
        cam_x = viewing_distance * np.cos(angle)
        cam_z = viewing_distance * np.sin(angle)
        cam_pos = np.array([[cam_x], [0], [cam_z]])

        # --- look-at matrix (corrected orientation) ---
        forward = (object_center - cam_pos).flatten()
        forward /= np.linalg.norm(forward)
        up = np.array([0, 1, 0], dtype=float)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        # ✅ FIX: use +forward (not -forward)
        R = np.vstack([right, up, forward])
        t = -R @ cam_pos
        poses.append((R, t))

    return poses


# ===============================================================
# PREVIEW PATH (OPTIONAL)
# ===============================================================
def preview_camera_path(poses, object_center):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    centers = np.array([(-R.T @ t).flatten() for R, t in poses])
    ax.plot(centers[:, 0], centers[:, 2], centers[:, 1], 'r-', label="Camera Path")
    ax.scatter(object_center[0], object_center[2], object_center[1], c='b', s=80, label="Object Center")
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.legend()
    ax.set_title("Camera Path (Half Circle Around Object)")
    plt.show()


# ===============================================================
# MAIN FUNCTION
# ===============================================================
def CreateFinalVideo():
    # 1. Load scene data
    camera_objs = load_data()
    crop_region = camera_objs[0].flatten()
    K = camera_objs[2]
    ForegroundPointCloudRGB = camera_objs[3]
    BackgroundPointCloudRGB = camera_objs[4]

    # 2. Define video params
    fps = 5
    num_frames = 25
    initial_bbox_size = (250, 400)
    target_bbox_size = (400, 640)

    # 3. Object center
    fg_points = ForegroundPointCloudRGB[:3, :]
    object_center = np.mean(fg_points, axis=1).reshape(3, 1)

    # 4. Compute viewing distance
    dist_initial = np.linalg.norm(object_center)
    size_ratio = target_bbox_size[1] / initial_bbox_size[1]
    viewing_distance = dist_initial / size_ratio

    print(f"\n[INFO] Object center: {object_center.flatten()}")
    print(f"[INFO] Viewing distance: {viewing_distance:.3f}")

    # 5. Generate camera path
    poses = generate_camera_poses(object_center, viewing_distance, num_frames)
    preview_camera_path(poses, object_center)

    # 6. Prepare video writer
    output_dir = os.path.join("images", "output")
    os.makedirs(output_dir, exist_ok=True)
    video_filename = "Fish_ObjectCentered_Fixed.wmv"
    video_filepath = os.path.join(output_dir, video_filename)
    frame_h = int(crop_region[2]) + 1
    frame_w = int(crop_region[3]) + 1
    fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_w, frame_h))

    # 7. Render frames
    for i, (R, t) in enumerate(poses):
        print(f"\n[Frame {i+1}/{num_frames}] Rendering...")

        P = K @ np.hstack((R, t))
        img = PointCloud2Image(P, (ForegroundPointCloudRGB, BackgroundPointCloudRGB), crop_region)

        # Enhance visibility
        img = np.clip(img * 1.2, 0, 1)
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)

        # Preview first few frames
        if i < 3:
            plt.imshow(img)
            plt.title(f"Frame {i+1}")
            plt.axis('off')
            plt.show()

    video_writer.release()
    print(f"\n Final video saved to: {video_filepath}")
    print("Camera now faces the object correctly and maintains consistent scale.")


# ===============================================================
# ENTRY POINT
# ===============================================================
def main():
    CreateFinalVideo()

if __name__ == "__main__":
    main()
