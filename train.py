import os
import tensorflow as tf
from model.pointnet import pointnet_model  
from utils.dataloader import get_datasets

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 16
NUM_CLASSES = 2  # Cube = 0, Sphere = 1
EPOCHS = 50
LEARNING_RATE = 0.0001

# -----------------------------
# Paths
# -----------------------------
HDF5_FILE_PATH = "pointclouds_preprocessed.h5"  
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# Training Function
# -----------------------------
def train():
    # Step 1: Load datasets (now includes validation set)
    train_dataset, val_dataset, test_dataset = get_datasets(HDF5_FILE_PATH, BATCH_SIZE)

    # Step 2: Define the PointNet model
    model = pointnet_model(num_classes=NUM_CLASSES)  # Uses default num_points=1024

    # Step 3: Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 4: Set up callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1
    )

    # Step 5: Train the model using validation data
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,  # <-- Now uses validation set
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )

    # Step 6: Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Step 7: Save the final model
    model.save(os.path.join(CHECKPOINT_DIR, "final_model.h5"))
    print("Training complete. Model saved.")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    train()