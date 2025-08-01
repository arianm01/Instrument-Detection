import os

import keras
import numpy as np
import pandas as pd
from keras.saving.save import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from tqdm import tqdm

import librosa
from src.utility.utils import balance_dataset_with_augmentation


def load_signal(file_path, SAMPLE_RATE):
    signal = librosa.load(file_path,
                          sr=SAMPLE_RATE)[0]
    return signal


def contains(main_string, substring):
    return substring in main_string


def get_files(instrument, folder):
    try:
        with open(folder + '/' + instrument + '.txt', 'r') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print("File not found.")
        return []


def read_data(dataset_path, merge_factor, duration=1, n_mfcc=26, n_fft=2048, hop_length=512,
              folder='./Models/split/train',
              balance_needed=True):
    """
    Reads audio files from the dataset directory, computes MFCC features, and returns them with labels.
    Optionally merges multiple audio samples into one data point based on the merge factor.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        duration (float): Duration of audio clips for processing (in seconds).
        n_mfcc (int): Number of MFCCs to return.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        merge_factor (int): Number of audio samples to merge into one data point.
        folder (str): Path to the folder
        balance_needed (bool): Whether to balance the data points
    Returns:
        tuple: Tuple containing:
            - x (list of np.array): MFCCs of each audio file (or merged files).
            - y (np.array): One-hot encoded class labels.
            - classes (list): List of class names.
    """
    x, y = [], []
    sample_rate = 22050
    # classes = ['Tar', 'Kamancheh', 'Santur', 'Setar', 'Ney']
    classes = os.listdir(dataset_path)
    print(classes)
    for i, instrument in enumerate(classes):
        print(instrument)
        files = get_files(instrument, folder)
        files.sort()
        process_files(files, dataset_path, instrument, merge_factor, duration, n_mfcc, n_fft, hop_length, x, y, i,
                      duration * merge_factor * sample_rate, int(duration * sample_rate))

    y = np.array(y)
    x = np.array(x)

    target_count = max(np.bincount(y))  # Adjust target count as needed
    print(target_count)
    if balance_needed:
        x, y = balance_dataset_with_augmentation(x, y, 22050, target_count)
    y = np.array(pd.get_dummies(y))
    return x, y, classes


def process_files(files, dataset_path, instrument, merge_factor, duration, n_mfcc, n_fft, hop_length, x, y, label,
                  window_size, step_size):
    base_signal, seg, last_file = [], 1, ''

    # List the file names once
    file_names = os.listdir(os.path.join(dataset_path, instrument))

    def load_file(file_name):
        """
        Check if the file_name starts with any of the prefixes in `files`.
        If so, loads the audio and returns the updated last_file, sample_rate, and appended signal.
        """
        for prefix in files:
            if file_name.startswith(prefix):
                file_path = os.path.join(dataset_path, instrument, file_name)
                signal, sample_rate = librosa.load(file_path, duration=duration)
                return file_name, signal, sample_rate
        return None, None, None

    for file_name in tqdm(file_names):
        # If the current file is part of the same group as the last file, accumulate the signal
        if contains(file_name, last_file[:-9]):
            new_last, signal, sample_rate = load_file(file_name)
            if signal is not None:
                base_signal.append(signal)
                last_file = new_last
                seg += 1
        # Otherwise, if there is accumulated data, process it and start a new accumulation
        elif base_signal:
            # Process the accumulated signals
            concatenated_signal = np.concatenate(base_signal)
            process_base_signal(concatenated_signal, sample_rate, merge_factor, window_size, step_size,
                                n_mfcc, n_fft, hop_length, x, y, label)
            # Reset accumulation
            base_signal, seg, last_file = [], 1, ""
            # Also load the current file to start a new accumulation
            new_last, signal, sample_rate = load_file(file_name)
            if signal is not None:
                base_signal.append(signal)
                last_file = new_last

    # Process any remaining accumulated signals after looping
    if base_signal:
        concatenated_signal = np.concatenate(base_signal)
        process_base_signal(concatenated_signal, sample_rate, merge_factor, window_size, step_size,
                            n_mfcc, n_fft, hop_length, x, y, label)


def process_base_signal(signal, sample_rate, merge_factor, window_size, step_size, n_mfcc, n_fft, hop_length, x, y,
                        label):
    # Create sliding windows with the specified merge_factor
    windows = create_sliding_windows(signal, window_size, step_size)

    for i, window in enumerate(windows):
        mel_spectrogram = compute_mel_spectrogram(window, sample_rate)
        x.append(mel_spectrogram)
        y.append(label)


def create_sliding_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows

    # MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    # mfccs = np.mean(MFCCs.T, axis=0)

    # Extract Chromagram
    # chromagram = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    # chromagram = np.mean(chromagram.T, axis=0)

    # Extract Spectral Contrast
    # spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate)
    # spectral_contrast = np.mean(spectral_contrast.T, axis=0)

    # Extract Tonnetz
    # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sample_rate)
    # tonnetz = np.mean(tonnetz.T, axis=0)
    # print(tonnetz.shape)

    # save_spectrogram_image( spectrogram, 3)

    # Combine all features into a single array
    # features = np.concatenate([chromagram, spectral_contrast])


def compute_mel_spectrogram(signal, sample_rate):
    spectrogram = extract_spectrogram(signal, sample_rate, n_mels=64)

    return spectrogram.T


def save_spectrogram_image(spectrogram, save_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def extract_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=64):
    # Compute the spectrogram
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Convert to mel scale
    mel_spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels)
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="1 sec accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="1 sec error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix shape:", cm.shape)
    print("Number of labels:", len(classes))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()

#
# def create_tsne_visualization(model, x_test, y_test, instrument_names, n_samples=10000):
#     """
#     Simple function to create t-SNE visualization of your model's features.
#
#     Args:
#         model: Your trained Keras model
#         x_test: Test data
#         y_test: Test labels (numeric or one-hot)
#         instrument_names: List of instrument names
#         n_samples: Number of samples to visualize
#     """
#
#     # Sample random test data
#     print(f"Sampling {n_samples} test segments for t-SNE visualization...")
#     np.random.seed(42)
#
#     if len(x_test) > n_samples:
#         indices = np.random.choice(len(x_test), size=n_samples, replace=False)
#         x_sampled = x_test[indices]
#         y_sampled = y_test[indices]
#     else:
#         x_sampled = x_test
#         y_sampled = y_test
#
#     # Convert one-hot to numeric if needed
#     if y_sampled.ndim > 1 and y_sampled.shape[1] > 1:
#         y_numeric = np.argmax(y_sampled, axis=1)
#     else:
#         y_numeric = y_sampled
#
#     # Find penultimate layer (layer before final classification layer)
#     penultimate_layer = None
#     for i, layer in enumerate(model.layers):
#         if 'dense' in layer.name.lower() and i < len(model.layers) - 1:
#             penultimate_layer = layer
#             break
#
#     if penultimate_layer is None:
#         penultimate_layer = model.layers[-2]  # Use second-to-last layer
#
#     print(f"Extracting features from layer: {penultimate_layer.name}")
#
#     # Create feature extraction model
#     from tensorflow import keras
#     feature_model = keras.Model(
#         inputs=model.input,
#         outputs=penultimate_layer.output
#     )
#
#     # Extract features
#     print("Extracting penultimate layer features...")
#     features = feature_model.predict(x_sampled, batch_size=32, verbose=0)
#     print(f"Feature shape: {features.shape}")
#
#     # Apply t-SNE
#     print("Applying t-SNE (this may take a few minutes)...")
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
#     embeddings_2d = tsne.fit_transform(features)
#
#     # Create visualization
#     plt.figure(figsize=(14, 10))
#
#     # Colors for 15 instruments
#     colors = plt.cm.tab20(np.linspace(0, 1, len(instrument_names)))
#
#     # Plot each instrument class
#     for i, instrument in enumerate(instrument_names):
#         mask = y_numeric == i
#         if np.any(mask):
#             plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
#                         c=[colors[i]], label=instrument, alpha=0.7, s=25,
#                         edgecolors='white', linewidth=0.5)
#
#     plt.xlabel('t-SNE Dimension 1', fontsize=12)
#     plt.ylabel('t-SNE Dimension 2', fontsize=12)
#     plt.title('t-SNE Visualization of Penultimate Layer Features\nPersian Musical Instruments', fontsize=14)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#
#     # Save the visualization
#     plt.savefig('tsne_persian_instruments.png', dpi=300, bbox_inches='tight')
#     plt.savefig('tsne_persian_instruments.pdf', bbox_inches='tight')
#     print("t-SNE visualization saved as 'tsne_persian_instruments.png' and '.pdf'")
#
#     plt.show()
#
#     return embeddings_2d, features
#
#
# # Alternative: Extract features from encoder model directly
# def create_tsne_from_encoder(encoder_path, x_test, y_test, instrument_names, n_samples=10000):
#     """
#     Create t-SNE visualization using encoder model directly.
#     Use this if you have separate encoder files.
#     """
#     from tensorflow import keras
#
#     # Load encoder
#     encoder = keras.models.load_model(encoder_path)
#
#     # Sample data
#     if len(x_test) > n_samples:
#         indices = np.random.choice(len(x_test), size=n_samples, replace=False)
#         x_sampled = x_test[indices]
#         y_sampled = y_test[indices]
#     else:
#         x_sampled = x_test
#         y_sampled = y_test
#
#     # Convert labels if needed
#     if y_sampled.ndim > 1:
#         y_numeric = np.argmax(y_sampled, axis=1)
#     else:
#         y_numeric = y_sampled
#
#     # Extract features from encoder
#     features = encoder.predict(x_sampled, batch_size=32)
#
#     # Apply t-SNE and visualize (same as above)
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
#     embeddings_2d = tsne.fit_transform(features)
#
#     # Create plot (same plotting code as above)
#     plt.figure(figsize=(14, 10))
#     colors = plt.cm.tab20(np.linspace(0, 1, len(instrument_names)))
#
#     for i, instrument in enumerate(instrument_names):
#         mask = y_numeric == i
#         if np.any(mask):
#             plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
#                         c=[colors[i]], label=instrument, alpha=0.7, s=25)
#
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.title('t-SNE Visualization from Encoder Features\nPersian Musical Instruments')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('tsne_encoder_features.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#     return embeddings_2d, features


def extract_and_visualize_features(model, X_test, y_test, instrument_names, n_samples=10000):
    """
    Extract features and create t-SNE visualization specific to your model architecture.
    """

    # Sample random test segments
    print(f"Sampling {n_samples} random test segments...")
    np.random.seed(42)

    if len(X_test) > n_samples:
        indices = np.random.choice(len(X_test), size=n_samples, replace=False)
        X_sampled = X_test[indices]
        y_sampled = y_test[indices]
    else:
        X_sampled = X_test
        y_sampled = y_test

    # Convert one-hot to numeric labels if needed
    if y_sampled.ndim > 1:
        y_numeric = np.argmax(y_sampled, axis=1)
    else:
        y_numeric = y_sampled

    print(f"Sampled data shape: {X_sampled.shape}")
    print(f"Unique classes in sample: {np.unique(y_numeric)}")

    # Extract penultimate layer features
    print("Extracting penultimate layer features...")

    # Method 1: Extract from penultimate dense layer
    penultimate_layer = None
    for i, layer in enumerate(model.layers):
        if 'dense' in layer.name.lower() and i < len(model.layers) - 1:
            penultimate_layer = layer

    if penultimate_layer is None:
        # Use the layer before the last one
        penultimate_layer = model.layers[-2]

    print(f"Using layer: {penultimate_layer.name} with output shape: {penultimate_layer.output_shape}")

    # Create feature extraction model
    feature_model = keras.Model(
        inputs=model.input,
        outputs=penultimate_layer.output
    )

    # Extract features
    features = feature_model.predict(X_sampled, batch_size=32, verbose=1)
    print(f"Extracted features shape: {features.shape}")

    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(features)

    # Create visualization
    create_persian_instrument_visualization(embeddings_2d, y_numeric, instrument_names)

    return embeddings_2d, features


def create_persian_instrument_visualization(embeddings_2d, labels, instrument_names):
    """
    Create a publication-ready t-SNE visualization for Persian instruments.
    """
    plt.figure(figsize=(16, 12))

    # Use distinct colors for better visualization
    colors = plt.cm.tab20(np.linspace(0, 1, len(instrument_names)))

    # Create the plot
    for i, instrument in enumerate(instrument_names):
        mask = labels == i
        if np.any(mask):
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=instrument,
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title('t-SNE Visualization of Penultimate Layer Features\nPersian Classical Musical Instruments',
              fontsize=16, pad=20)

    # Create legend with better formatting
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig('tsne_persian_instruments.png', dpi=300, bbox_inches='tight')
    plt.savefig('tsne_persian_instruments.pdf', bbox_inches='tight')  # For publication

    print("Visualization saved as 'tsne_persian_instruments.png' and 'tsne_persian_instruments.pdf'")
    plt.show()

    # Print some analysis
    analyze_clusters(embeddings_2d, labels, instrument_names)


def analyze_clusters(embeddings_2d, labels, instrument_names):
    """
    Analyze cluster separation and provide insights.
    """
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS")
    print("=" * 60)

    # Calculate centroids
    centroids = {}
    cluster_sizes = {}

    for i, instrument in enumerate(instrument_names):
        mask = labels == i
        if np.any(mask):
            centroids[instrument] = np.mean(embeddings_2d[mask], axis=0)
            cluster_sizes[instrument] = np.sum(mask)
            print(f"{instrument}: {cluster_sizes[instrument]} samples")

    print(f"\nTotal samples visualized: {len(labels)}")

    # Identify well-separated vs. overlapping clusters
    print("\nCluster Separation Analysis:")
    print("-" * 30)

    # Calculate pairwise distances between centroids
    from scipy.spatial.distance import pdist, squareform
    centroid_names = list(centroids.keys())
    centroid_positions = np.array(list(centroids.values()))

    if len(centroid_positions) > 1:
        distances = squareform(pdist(centroid_positions))

        # Find most separated pairs
        max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
        print(f"Most separated: {centroid_names[max_dist_idx[0]]} - {centroid_names[max_dist_idx[1]]}")

        # Find closest pairs (excluding diagonal)
        np.fill_diagonal(distances, np.inf)
        min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
        print(f"Closest clusters: {centroid_names[min_dist_idx[0]]} - {centroid_names[min_dist_idx[1]]}")

    print("\nExpected findings (based on timbral similarity):")
    print("- Well-separated: Percussion (Daf, Tonbak) vs. Strings vs. Winds")
    print("- Partially overlapping: Similar strings (Setar, Tar, Oud)")
    print("- Wind instruments: Ney vs. Ney Anban separation")