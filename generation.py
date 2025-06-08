import os
import numpy as np
import pybullet as p
from tqdm import tqdm

# --------------------------
# Parameters
# --------------------------
DATASET_DIR = "data"  
NUM_TRAIN_SAMPLES = 2000
NUM_VAL_SAMPLES = 450
NUM_TEST_SAMPLES = 500
TOTAL_SAMPLES = NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES + NUM_TEST_SAMPLES

# --------------------------
# Helper Functions
# --------------------------

def depth_to_point_cloud(depth_image, width=640, height=480):
    far = 100.0
    near = 0.01
    true_depth = (far * near) / (far - (far - near) * depth_image)
    points = []
    fx = width / 2
    fy = height / 2

    for h in range(height):
        for w in range(width):
            z = true_depth[h][w]
            if z > 2.5:
                continue
            x = (w - fx) * z / fx
            y = (fy - h) * z / fy
            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def get_top_down_point_cloud(position=[0, 0, 1], yaw=0, pitch=-89.9, roll=0):
    # Randomized camera orientation
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 1],
        distance=2,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(60, 640/480, 0.1, 100)

    images = p.getCameraImage(640, 480, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    depth_img = np.reshape(images[3], [480, 640])
    return depth_to_point_cloud(depth_img)


def randomize_pose():
    x = np.random.uniform(-0.2, 0.2)
    y = np.random.uniform(-0.2, 0.2)
    yaw = np.random.uniform(0, 360)
    return [x, y, 1], np.radians(yaw)


def randomize_camera_angles():
    pitch = np.random.uniform(-89, -60)
    yaw = np.random.uniform(0, 360)
    roll = np.random.uniform(-10, 10)
    return np.radians(pitch), np.radians(yaw), np.radians(roll)


def save_point_cloud(path, pc):
    np.save(path, pc)


# --------------------------
# Setup Dataset Folders
# --------------------------

os.makedirs(os.path.join(DATASET_DIR, "cube", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "cube", "val"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "cube", "test"), exist_ok=True)

os.makedirs(os.path.join(DATASET_DIR, "sphere", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "sphere", "val"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "sphere", "test"), exist_ok=True)

# --------------------------
# Start Simulation
# --------------------------

p.connect(p.DIRECT)  # Headless mode
p.setGravity(0, 0, -10)

# Collision shapes
cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)

# --------------------------
# Generate Cube Data
# --------------------------

print("Generating cube data...")
body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, basePosition=[0, 0, 1])

for i in tqdm(range(TOTAL_SAMPLES), desc="Cube"):
    pos, yaw = randomize_pose()
    pitch, cam_yaw, roll = randomize_camera_angles()

    p.resetBasePositionAndOrientation(body_id, posObj=pos, ornObj=p.getQuaternionFromEuler([0, 0, yaw]))
    for _ in range(10):
        p.stepSimulation()
    pc = get_top_down_point_cloud(pos, yaw=cam_yaw, pitch=pitch, roll=roll)

    # Save
    if i < NUM_TRAIN_SAMPLES:
        save_point_cloud(os.path.join(DATASET_DIR, "cube", "train", f"pointcloud_{i}.npy"), pc)
    elif i < NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES:
        save_point_cloud(os.path.join(DATASET_DIR, "cube", "val", f"pointcloud_{i - NUM_TRAIN_SAMPLES}.npy"), pc)
    else:
        save_point_cloud(os.path.join(DATASET_DIR, "cube", "test", f"pointcloud_{i - NUM_TRAIN_SAMPLES - NUM_VAL_SAMPLES}.npy"), pc)

p.removeBody(body_id)

# --------------------------
# Generate Sphere Data
# --------------------------

print("Generating sphere data...")
body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape, basePosition=[0, 0, 1])

for i in tqdm(range(TOTAL_SAMPLES), desc="Sphere"):
    pos, yaw = randomize_pose()
    pitch, cam_yaw, roll = randomize_camera_angles()

    p.resetBasePositionAndOrientation(body_id, posObj=pos, ornObj=p.getQuaternionFromEuler([0, 0, yaw]))
    for _ in range(10):
        p.stepSimulation()
    pc = get_top_down_point_cloud(pos, yaw=cam_yaw, pitch=pitch, roll=roll)

    # Save
    if i < NUM_TRAIN_SAMPLES:
        save_point_cloud(os.path.join(DATASET_DIR, "sphere", "train", f"pointcloud_{i}.npy"), pc)
    elif i < NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES:
        save_point_cloud(os.path.join(DATASET_DIR, "sphere", "val", f"pointcloud_{i - NUM_TRAIN_SAMPLES}.npy"), pc)
    else:
        save_point_cloud(os.path.join(DATASET_DIR, "sphere", "test", f"pointcloud_{i - NUM_TRAIN_SAMPLES - NUM_VAL_SAMPLES}.npy"), pc)

p.disconnect()

print(f"- Train: {NUM_TRAIN_SAMPLES} per class")
print(f"- Val: {NUM_VAL_SAMPLES} per class")
print(f"- Test: {NUM_TEST_SAMPLES} per class")