import pybullet as p
import numpy as np
import tensorflow as tf

# -----------------------------
# Helper Functions
# -----------------------------

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


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points


# -----------------------------
# Set up PyBullet Simulation
# -----------------------------
p.connect(p.GUI)
p.setGravity(0, 0, -10)

# Create a cube
cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
cube_body = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=cube_shape,
    basePosition=[0, 0, 1]
)

# Step simulation so everything settles
for _ in range(10):
    p.stepSimulation()

# -----------------------------
# Set up camera (top-down view)
# -----------------------------
width, height = 640, 480

view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 1],
    distance=2,
    yaw=20,
    pitch=-65.9,
    roll=20,
    upAxisIndex=2
)

proj_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100)

# Get camera image
images = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
depth_img = np.reshape(images[3], [height, width])

# Convert to point cloud
point_cloud = depth_to_point_cloud(depth_img, width, height)

# Sample exactly 1024 points
NUM_POINTS = 1024
if len(point_cloud) < NUM_POINTS:
    choice = np.random.choice(len(point_cloud), NUM_POINTS, replace=True)
else:
    choice = np.random.choice(len(point_cloud), NUM_POINTS, replace=False)
sampled_points = point_cloud[choice]

# Normalize
sampled_points = normalize_point_cloud(sampled_points)

# Add batch dimension and convert to float32
input_point_cloud = np.expand_dims(sampled_points, axis=0).astype(np.float32)

# -----------------------------
# Load model and predict
# -----------------------------
model = tf.keras.models.load_model("checkpoints/best_model.h5")
pred = model.predict(input_point_cloud)
predicted_class = np.argmax(pred, axis=1)[0]

# -----------------------------
# Print result
# -----------------------------
class_names = ["Cube", "Sphere"]
print("Predicted class:", predicted_class)
print("Class name:", class_names[predicted_class])
print("Raw probabilities:", pred)