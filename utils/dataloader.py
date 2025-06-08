import h5py
import tensorflow as tf
import numpy as np

def load_h5_data(file_path):
    """
    Load preprocessed data from an HDF5 file.
    Returns train, val, and test splits.
    """
    with h5py.File(file_path, 'r') as f:
        train_points = f['train_points'][:].astype(np.float32)
        train_labels = f['train_labels'][:].astype(np.int64)
        val_points = f['val_points'][:].astype(np.float32)
        val_labels = f['val_labels'][:].astype(np.int64)
        test_points = f['test_points'][:].astype(np.float32)
        test_labels = f['test_labels'][:].astype(np.int64)

    return (train_points, train_labels), (val_points, val_labels), (test_points, test_labels)


def augment_point_cloud(points):
    """
    Apply data augmentation to a point cloud.
    - Random rotation around the z-axis
    - Random jittering (Gaussian noise)
    """
    points = tf.cast(points, tf.float32)

    # Random rotation around Z-axis
    theta = tf.random.uniform((), minval=0, maxval=2 * np.pi)
    rotation_matrix = tf.stack([
        [tf.cos(theta), -tf.sin(theta), 0],
        [tf.sin(theta), tf.cos(theta), 0],
        [0, 0, 1]
    ])
    points = tf.matmul(points, rotation_matrix)

    # Random jittering (Gaussian noise)
    jitter = tf.random.normal(tf.shape(points), mean=0.0, stddev=0.01)
    points += jitter

    return points


def create_dataset(points, labels, batch_size, augment=False, shuffle_buffer=None):
    """
    Create a TensorFlow Dataset pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices((points, labels))

    if shuffle_buffer is not None:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True
        )

    if augment:
        dataset = dataset.map(
            lambda x, y: (augment_point_cloud(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_datasets(file_path, batch_size):
    """
    Load and create training, validation, and testing datasets.
    """
    (train_points, train_labels), (val_points, val_labels), (test_points, test_labels) = load_h5_data(file_path)

    train_dataset = create_dataset(
        train_points, train_labels, batch_size,
        augment=True, shuffle_buffer=len(train_points)
    )

    val_dataset = create_dataset(
        val_points, val_labels, batch_size,
        augment=False, shuffle_buffer=None
    )

    test_dataset = create_dataset(
        test_points, test_labels, batch_size,
        augment=False, shuffle_buffer=None
    )

    return train_dataset, val_dataset, test_dataset