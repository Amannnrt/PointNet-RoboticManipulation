import os
import numpy as np
import h5py
from tqdm import tqdm

def normalize_point_cloud(points):
    """
    Center and scale point cloud to fit inside a unit sphere.
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points


def preprocess_pointclouds(root_dir, output_file, num_points=1024):
    """
    Preprocess .npy point clouds from folder structure like:

    root_dir/
        cube/
            train/
                pointcloud_0.npy ...
            val/
            test/
        sphere/
            train/
            val/
            test/

    Outputs: HDF5 file with normalized point clouds and labels
    """
    categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {cat: i for i, cat in enumerate(categories)}  # e.g. {'cube': 0, 'sphere': 1}

    all_train_points = []
    all_train_labels = []
    all_val_points = []
    all_val_labels = []
    all_test_points = []
    all_test_labels = []

    for cat in categories:
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(root_dir, cat, split)
            if not os.path.exists(split_dir):
                print(f"Skipping missing directory: {split_dir}")
                continue

            files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
            for file_name in tqdm(files, desc=f"{split.upper()} {cat}"):
                file_path = os.path.join(split_dir, file_name)
                points = np.load(file_path)

                # Sample or pad to fixed number of points
                if len(points) < num_points:
                    choice = np.random.choice(len(points), num_points, replace=True)
                else:
                    choice = np.random.choice(len(points), num_points, replace=False)
                sampled_points = points[choice]

                # Normalize
                sampled_points = normalize_point_cloud(sampled_points)

                if split == "train":
                    all_train_points.append(sampled_points)
                    all_train_labels.append(label_map[cat])
                elif split == "val":
                    all_val_points.append(sampled_points)
                    all_val_labels.append(label_map[cat])
                else:
                    all_test_points.append(sampled_points)
                    all_test_labels.append(label_map[cat])

    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train_points', data=np.array(all_train_points))
        f.create_dataset('train_labels', data=np.array(all_train_labels))
        f.create_dataset('val_points', data=np.array(all_val_points))
        f.create_dataset('val_labels', data=np.array(all_val_labels))
        f.create_dataset('test_points', data=np.array(all_test_points))
        f.create_dataset('test_labels', data=np.array(all_test_labels))

    print("Preprocessing complete. Saved to:", output_file)


if __name__ == "__main__":
    root_dir = "data"  # Folder containing cube/ and sphere/
    output_file = "pointclouds_preprocessed.h5"
    preprocess_pointclouds(root_dir, output_file)