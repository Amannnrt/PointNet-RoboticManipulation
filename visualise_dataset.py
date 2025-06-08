import os
import numpy as np
import open3d as o3d

# -------------------------------
# Configuration
# -------------------------------
NUM_SAMPLES_TO_SHOW = 10
DATASET_ROOT = "data"  # Folder containing 'cube' and 'sphere'
DOWNSAMPLE_POINTS = 1000

# Define paths
cube_train_dir = os.path.join(DATASET_ROOT, "cube", "train")
sphere_train_dir = os.path.join(DATASET_ROOT, "sphere", "train")

# Function to load and visualize point cloud
def visualize_point_cloud(points, title="Point Cloud", color=[1, 0, 0]):
    # Downsample
    if len(points) > DOWNSAMPLE_POINTS:
        indices = np.random.choice(len(points), DOWNSAMPLE_POINTS, replace=False)
        points = points[indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)

    o3d.visualization.draw_geometries([pcd],
                                     window_name=title,
                                     width=800,
                                     height=600,
                                     point_show_normal=False)


# -------------------------------
# Load and visualize cubes
# -------------------------------
cube_files = [os.path.join(cube_train_dir, f) for f in os.listdir(cube_train_dir) if f.endswith(".npy")]
print("\n--- Visualizing Cubes ---")
for i, filepath in enumerate(cube_files[:NUM_SAMPLES_TO_SHOW]):
    points = np.load(filepath)
    print(f"Visualizing Cube {i+1}: {os.path.basename(filepath)}")
    visualize_point_cloud(points, title=f"Cube {i+1}", color=[1, 0, 0])  # Red


# -------------------------------
# Load and visualize spheres
# -------------------------------
sphere_files = [os.path.join(sphere_train_dir, f) for f in os.listdir(sphere_train_dir) if f.endswith(".npy")]
print("\n--- Visualizing Spheres ---")
for i, filepath in enumerate(sphere_files[:NUM_SAMPLES_TO_SHOW]):
    points = np.load(filepath)
    print(f"Visualizing Sphere {i+1}: {os.path.basename(filepath)}")
    visualize_point_cloud(points, title=f"Sphere {i+1}", color=[0, 0, 1])  # Blue


print("\n Visualization complete.")