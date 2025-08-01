import librosa
import numpy as np
import tensorflow as tf
from keras.saving.save import load_model


def test_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    tf.config.list_physical_devices('GPU')


def reshape_data(data, time_steps):
    num_samples = data.shape[0]
    num_frames = int(data.shape[1] / time_steps) * time_steps
    num_mfcc = 13
    num_channels = 1
    frames_per_step = num_frames // time_steps  # This should be 43 time steps

    # Reshape the data
    new_shape = (num_samples, time_steps, frames_per_step, num_mfcc, num_channels)

    return np.reshape(data[:, :num_frames, :, :], new_shape)


def pitch_shift(data, sr, n_steps):
    shifted_data = librosa.effects.pitch_shift(data.T, sr=sr, n_steps=n_steps).T
    return shifted_data


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(*data.shape)
    augmented_data = data + noise_factor * noise
    return augmented_data


def time_shift(data, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * data.shape[1])
    shifted_data = np.roll(data, shift, axis=1)
    return shifted_data


def change_volume(data, volume_change=0.5):
    return data * volume_change


def augment_data(data, sr, label, target_count):
    augmented_data = []
    augmented_labels = []
    current_count = len(data)

    while current_count < target_count:
        sample_idx = np.random.randint(len(data))
        sample = data[sample_idx]

        # Choose an augmentation
        aug_type = np.random.choice(['pitch_shift', 'add_noise', 'time_shift', 'change_volume'])

        if aug_type == 'pitch_shift':
            augmented_sample = pitch_shift(sample, sr, n_steps=np.random.randint(-5, 5))
        elif aug_type == 'add_noise':
            augmented_sample = add_noise(sample, noise_factor=0.005)
        elif aug_type == 'time_shift':
            augmented_sample = time_shift(sample, shift_max=0.2)
        elif aug_type == 'change_volume':
            augmented_sample = change_volume(sample, volume_change=np.random.uniform(0.5, 1.5))

        augmented_data.append(augmented_sample)
        augmented_labels.append(label)
        current_count += 1

    return np.array(augmented_data), np.array(augmented_labels)


def balance_dataset_with_augmentation(x, y, sr, target_count, models=None):
    unique_classes = np.unique(y)
    balanced_x = []
    balanced_y = []

    for cls in unique_classes:
        class_indices = np.nonzero(y == cls)[0].astype(np.int32)
        # Debug information
        print(f"Class: {cls}")
        print(f"Class Indices: {class_indices}")
        print(f"Class Indices Type: {type(class_indices)}")
        print(f"Class Indices Dtype: {class_indices.dtype}")
        print(f"x Shape: {x.shape}")
        class_data = x[class_indices]
        class_labels = y[class_indices]

        if len(class_data) < target_count:
            augmented_data, augmented_labels = augment_data(class_data, sr, cls, target_count)
            class_data = np.concatenate((class_data, augmented_data))
            class_labels = np.concatenate((class_labels, augmented_labels))
        if models:
            class_data = get_meta_features(models, class_data.astype(np.float32)[..., np.newaxis])
        balanced_x.append(class_data.astype(np.float32))
        balanced_y.append(class_labels)

    balanced_y = np.concatenate(balanced_y)
    balanced_x = np.concatenate(balanced_x)

    return balanced_x, balanced_y


def create_label_mapping(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    print(label_to_index, index_to_label)
    return label_to_index, index_to_label


def convert_labels_to_indices(labels, label_to_index):
    return np.array([label_to_index[label] for label in labels])


def sanitize_file_name(file_name):
    """Sanitize the file name to avoid invalid characters."""
    return "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in file_name])


def split_into_chunks(X, chunk_size):
    """
    Splits the input array X into chunks of size chunk_size along axis 1.
    Pads the array if necessary to make the split possible.

    Parameters:
    - X: np.ndarray, shape (batch_size, dim1, dim2)
    - chunk_size: int, size of each chunk along axis 1
    - padding_shape: tuple, padding to apply (only along axis 1)

    Returns:
    - List of np.ndarray chunks
    """
    # Calculate the number of chunks needed
    dim1 = X.shape[1]
    remainder = dim1 % chunk_size
    padding_needed = (chunk_size - remainder) if remainder != 0 else 0

    if padding_needed > 0:
        # Define pad_width dynamically based on X's dimensions
        pad_width = [(0, 0)] * X.ndim  # Initialize pad_width for all dimensions
        pad_width[1] = (0, padding_needed)  # Pad only along axis 1

        # Apply padding
        X_padded = np.pad(X, pad_width, mode='constant', constant_values=0)
    else:
        X_padded = X

    # Now split into chunks
    num_chunks = X_padded.shape[1] // chunk_size
    chunks = np.split(X_padded, num_chunks, axis=1)

    return chunks


def get_model_feature(x, models=None):
    if models is None:
        model_paths = ['./Models/model_best_Tar.h5', './Models/model_best_Kamancheh.h5',
                       './Models/model_best_Santur.h5', './Models/model_best_Setar.h5',
                       './Models/model_best_Ney.h5']
        models = [load_model(model_paths[0]), load_model(model_paths[1]), load_model(model_paths[2]),
                  load_model(model_paths[3]), load_model(model_paths[4])]
    meta_features = get_meta_features(models, x)
    return meta_features


def get_meta_features(models, X, chunk_size=44):
    """Generate meta-features using predictions from base models."""
    chunks = split_into_chunks(X, chunk_size)
    all_features = []
    num_segments = chunks[0].shape[0]
    chunk_size = 1000
    for chunk in chunks:
        chunk_pred = []
        for start in range(0, num_segments, chunk_size):
            end = min(start + chunk_size, num_segments)
            segment_chunk_1 = chunk[start:end]
            chunk_predictions_1 = np.concatenate([model.predict(segment_chunk_1) for model in models], axis=1)
            chunk_pred.append(chunk_predictions_1)
        chunk_pred = np.concatenate(chunk_pred, axis=0)  # Flatten the list of chunk predictions
        all_features.append(chunk_pred)

    # Concatenate features from all chunks
    meta_features = np.concatenate(all_features, axis=1)

    return meta_features
