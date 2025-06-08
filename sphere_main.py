import pybullet as p
import numpy as np
import tensorflow as tf
import time
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Configuration Constants
# -----------------------------
WIDTH, HEIGHT = 640, 480
NUM_POINTS = 1024
MODEL_PATH = "checkpoints/best_model.h5"

CLASS_NAMES = ["Other", "Sphere"]
SPHERE_RADIUS = 0.025
SPHERE_POSITION = [0.5, 0.0, SPHERE_RADIUS]
TRAY_POSITION = [0.7, -0.3, 0]
CAMERA_DISTANCE = 0.2
CAMERA_PITCH = -60

# -----------------------------
# Helper Functions 
# -----------------------------
def depth_to_point_cloud(depth_image, seg_image, object_id):
    far = 100.0
    near = 0.01
    true_depth = (far * near) / (far - (far - near) * depth_image)
    points = []
    fx = WIDTH / 2
    fy = HEIGHT / 2
    for h in range(HEIGHT):
        for w in range(WIDTH):
            if seg_image[h, w] != object_id:
                continue
            z = true_depth[h][w]
            if z > 1.0:
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

def sample_points(points, num_points):
    if len(points) < num_points:
        indices = np.random.choice(len(points), num_points, replace=True)
    else:
        indices = np.random.choice(len(points), num_points, replace=False)
    return points[indices]

def visualize_point_cloud(points):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
    ax.set_title("Captured Point Cloud")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# -----------------------------
# Simulation Setup
# -----------------------------
def setup_simulation():
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane.urdf")

    # Create smaller sphere
    sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS)
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=SPHERE_RADIUS, rgbaColor=[0, 0, 1, 1])  # Blue sphere
    sphere_body = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=sphere_shape,
        baseVisualShapeIndex=sphere_visual,
        basePosition=SPHERE_POSITION
    )

    # Create tray with walls
    tray_half_extents = [0.1, 0.1, 0.02]
    wall_thickness = 0.01
    wall_height = 0.1

    # Create base of the tray
    tray_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=tray_half_extents)
    tray_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=tray_half_extents, rgbaColor=[0.6, 0.8, 1.0, 1.0])
    tray_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=tray_col_id,
        baseVisualShapeIndex=tray_visual_id,
        basePosition=TRAY_POSITION
    )

    # Create 4 walls (left, right, front, back)
    wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tray_half_extents[0], wall_thickness, wall_height / 2])
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[tray_half_extents[0], wall_thickness, wall_height / 2], rgbaColor=[0.4, 0.4, 0.4, 1])

    # Left wall
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=wall_collision,
        baseVisualShapeIndex=wall_visual,
        basePosition=[TRAY_POSITION[0], TRAY_POSITION[1] + tray_half_extents[1] - wall_thickness, TRAY_POSITION[2] + wall_height / 2]
    )
    # Right wall
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=wall_collision,
        baseVisualShapeIndex=wall_visual,
        basePosition=[TRAY_POSITION[0], TRAY_POSITION[1] - tray_half_extents[1] + wall_thickness, TRAY_POSITION[2] + wall_height / 2]
    )
    # Front wall
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=wall_collision,
        baseVisualShapeIndex=wall_visual,
        basePosition=[TRAY_POSITION[0] + tray_half_extents[0] - wall_thickness, TRAY_POSITION[1], TRAY_POSITION[2] + wall_height / 2],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 1.57])
    )
    # Back wall
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=wall_collision,
        baseVisualShapeIndex=wall_visual,
        basePosition=[TRAY_POSITION[0] - tray_half_extents[0] + wall_thickness, TRAY_POSITION[1], TRAY_POSITION[2] + wall_height / 2],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 1.57])
    )

    # Load robotic arm
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)

    return robot_id, sphere_body, tray_id, plane_id

# -----------------------------
# Point Cloud Capture
# -----------------------------
def capture_point_cloud(object_id):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=SPHERE_POSITION,
        distance=CAMERA_DISTANCE,
        yaw=25,
        pitch=CAMERA_PITCH,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(70, WIDTH/HEIGHT, 0.01, 2.0)

    _, _, _, depth_buffer, seg_buffer = p.getCameraImage(
        WIDTH, HEIGHT, 
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )

    depth_img = np.reshape(depth_buffer, [HEIGHT, WIDTH])
    seg_img = np.reshape(seg_buffer, [HEIGHT, WIDTH])

    return depth_to_point_cloud(depth_img, seg_img, object_id)

# -----------------------------
# Prediction Pipeline
# -----------------------------
def run_prediction(point_cloud):
    sampled_points = sample_points(point_cloud, NUM_POINTS)
    normalized_points = normalize_point_cloud(sampled_points)
    input_data = np.expand_dims(normalized_points, axis=0).astype(np.float32)

    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions[0])
    scores = tf.nn.softmax(predictions[0]).numpy()

    return predicted_class, scores

# -----------------------------
# Robot Control Functions
# -----------------------------
def move_robot(robot_id, object_id, tray_id, predicted_class):
    if predicted_class != 1:
        print("Object is not a sphere. Skipping manipulation.")
        return
    print("Starting manipulation sequence...")

    grasp_height = SPHERE_POSITION[2] + SPHERE_RADIUS + 0.01
    approach_height = grasp_height + 0.05
    # Increased transit height to clear tray walls (0.1m high)
    transit_height = TRAY_POSITION[2] + 0.15  # Well above wall_height (0.1)
    tray_height = TRAY_POSITION[2] + 0.05
    tray_approach_height = tray_height + 0.05

    # Define waypoints to ensure clearance over tray walls
    targets = [
        [SPHERE_POSITION[0], SPHERE_POSITION[1], approach_height],  # Approach above sphere
        [SPHERE_POSITION[0], SPHERE_POSITION[1], grasp_height],     # Grasp position
        [SPHERE_POSITION[0], SPHERE_POSITION[1], transit_height],   # Lift high after grasping
        # Intermediate point to avoid walls
        [(SPHERE_POSITION[0] + TRAY_POSITION[0]) / 2, (SPHERE_POSITION[1] + TRAY_POSITION[1]) / 2, transit_height],
        [TRAY_POSITION[0], TRAY_POSITION[1], tray_approach_height], # Approach above tray
        [TRAY_POSITION[0], TRAY_POSITION[1], tray_height],          # Place position
        [TRAY_POSITION[0], TRAY_POSITION[1], transit_height],       # Lift high after placing
    ]

    gripper_link = 11
    move_stage = 0
    attached = False
    constraint_id = None
    gripper_open = [0.04, 0.04]  # Gripper fully open
    gripper_closed = [0.01, 0.01]  # Gripper closed for sphere

    # Set initial gripper state (open)
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_open[0], force=100)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_open[1], force=100)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1./240.)

    # Collision filter function
    def set_collision_filter(body_a, link_a, body_b, link_b, enable_collision):
        p.setCollisionFilterPair(body_a, body_b, link_a, link_b, int(enable_collision))

    print("[INFO] Starting robot motion...")
    while move_stage < len(targets):
        target_pos = targets[move_stage]
        joint_positions = p.calculateInverseKinematics(
            robot_id, gripper_link, target_pos,
            maxNumIterations=200,
            residualThreshold=0.001
        )
        for i in range(7):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL,
                joint_positions[i],
                force=400,
                maxVelocity=0.9
            )
        p.stepSimulation()
        time.sleep(1./240.)
        current_pos = p.getLinkState(robot_id, gripper_link)[0]
        dist = sum((current_pos[i] - target_pos[i]) ** 2 for i in range(3))
        if dist < 0.002:
            print(f"[STEP {move_stage}] Reached target.")
            if move_stage == 1 and not attached:
                print("[ATTACHING] Closing gripper...")
                p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_closed[0], force=100)
                p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_closed[1], force=100)
                for _ in range(50):
                    p.stepSimulation()
                    time.sleep(1./240.)
                print("[ATTACHING] Gripping the object...")
                constraint_id = p.createConstraint(
                    robot_id, gripper_link,
                    object_id, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, -SPHERE_RADIUS],
                    [0, 0, 0]
                )
                attached = True
                # Disable collision between fingers and object
                for finger_link in [9, 10]:
                    set_collision_filter(robot_id, finger_link, object_id, -1, False)
                move_stage = 2
            elif move_stage == 4 and attached:
                print("[RELEASING] Opening gripper...")
                p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_open[0], force=100)
                p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_open[1], force=100)
                for _ in range(50):
                    p.stepSimulation()
                    time.sleep(1./240.)
                print("[RELEASING] Letting go of the object...")
                p.removeConstraint(constraint_id)
                attached = False
                # Re-enable collision
                for finger_link in [9, 10]:
                    set_collision_filter(robot_id, finger_link, object_id, -1, True)
                print("[WAITING] Allowing object to settle...")
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(1./240.)
                move_stage = 5
            else:
                move_stage += 1

    print("[DONE] Object placed safely.")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    robot_id, sphere_id, tray_id, plane_id = setup_simulation()

    print("Allowing scene to settle...")
    for _ in range(150):
        p.stepSimulation()
        time.sleep(1./240.)

    print("Capturing point cloud...")
    point_cloud = capture_point_cloud(sphere_id)
    print(f"Captured point cloud with {len(point_cloud)} points")

    visualize_point_cloud(point_cloud)

    print("Running prediction...")
    predicted_class, scores = run_prediction(point_cloud)

    print("\n===== PREDICTION RESULTS =====")
    print(f"Predicted class: {predicted_class} ({CLASS_NAMES[predicted_class]})")
    for i, score in enumerate(scores):
        print(f"{CLASS_NAMES[i]}: {score:.4f}")

    move_robot(robot_id, sphere_id, tray_id, predicted_class)

    print("\nSimulation complete. Press 'Q' to exit")
    while True:
        p.stepSimulation()
        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break
        time.sleep(1./240.)
    p.disconnect()