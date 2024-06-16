import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from tqdm import tqdm

import librosa

from utility.utils import balance_dataset_with_augmentation, create_label_mapping, convert_labels_to_indices


def load_signal(file_path, SAMPLE_RATE):
    signal = librosa.load(file_path,
                          sr=SAMPLE_RATE)[0]
    return signal


def contains(main_string, substring):
    return substring in main_string


def read_data(dataset_path, merge_factor, duration=1, n_mfcc=13, n_fft=512, hop_length=512):
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

    Returns:
        tuple: Tuple containing:
            - x (list of np.array): MFCCs of each audio file (or merged files).
            - y (np.array): One-hot encoded class labels.
            - classes (list): List of class names.
    """
    x, y = [], []
    sample_rate = 16000
    classes = ['Tar', 'Kamancheh', 'Santur', 'Setar', 'Ney']
    print(classes)
    for i, instrument in enumerate(classes):
        print(instrument)
        files = os.listdir(os.path.join(dataset_path, str(instrument)))
        process_files(files, dataset_path, instrument, merge_factor, duration, n_mfcc, n_fft, hop_length, x, y, i,
                      5 * merge_factor * sample_rate, int(5 * merge_factor * sample_rate / 2))

    y = np.array(y)

    target_count = max(np.bincount(y))  # Adjust target count as needed
    print(target_count)
    # x, y = balance_dataset_with_augmentation(np.array(x), y, 16000, target_count)
    y = np.array(pd.get_dummies(y))

    print(x[0], len(y))
    return x, y, classes


def separate_and_balance_data(X, y, instruments):
    """Separate and balance data for each instrument."""
    instrument_data = {}
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    for y_label in np.unique(y_labels):
        # Binary labels: 1 for the instrument, 0 for others
        y_binary = (y_labels == y_label).astype(int)

        # Separate the positive and negative classes
        X_pos = X[y_binary == 1]
        X_neg = X[y_binary == 0]
        y_pos = y_binary[y_binary == 1]
        y_neg = y_binary[y_binary == 0]

        # Undersample the negative class
        X_neg_resampled, y_neg_resampled = resample(X_neg, y_neg, replace=False, n_samples=len(X_pos), random_state=42)

        # Combine the resampled negative class with the positive class
        X_balanced = np.vstack((X_pos, X_neg_resampled))
        y_balanced = np.hstack((y_pos, y_neg_resampled))

        instrument_data[instruments[y_label]] = (X_balanced, y_balanced)

    return instrument_data


def process_files(files, dataset_path, instrument, merge_factor, duration, n_mfcc, n_fft, hop_length, x, y, label,
                  window_size, step_size):
    base_signal, seg, last_file = [], 1, ''

    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(dataset_path, instrument, file)
        signal, sample_rate = librosa.load(file_path, duration=duration)

        # if i == 12000:
        #     break

        if not contains(file, last_file[:-9]):
            # Process the accumulated base_signal
            if base_signal:
                base_signal = np.concatenate(base_signal)
                process_base_signal(base_signal, sample_rate, merge_factor, window_size, step_size, n_mfcc, n_fft,
                                    hop_length, x, y, label)
            base_signal, seg = [], 1

        base_signal.append(signal)
        last_file = file
        seg += 1

    # Process the remaining signals after the loop
    if base_signal:
        base_signal = np.concatenate(base_signal)
        process_base_signal(base_signal, step_size, merge_factor, window_size, step_size, n_mfcc, n_fft, hop_length,
                            x, y, label)


def process_base_signal(signal, sample_rate, merge_factor, window_size, step_size, n_mfcc, n_fft, hop_length, x, y,
                        label):
    # Create sliding windows with the specified merge_factor
    windows = create_sliding_windows(signal, window_size, step_size)

    for i, window in enumerate(windows):
        mfcc = compute_mfcc(window, sample_rate, n_mfcc, n_fft, hop_length)
        x.append(mfcc)
        y.append(label)


def create_sliding_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows


def compute_mfcc(signal, sample_rate, n_mfcc, n_fft, hop_length):
    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    return normalize_mfccs(MFCCs).T  # Normalize MFCCs


def normalize_mfccs(mfccs):
    # Calculate mean and standard deviation of the MFCCs
    # mean = np.mean(mfccs, axis=0)
    # std = np.std(mfccs, axis=0)
    #
    # # Apply Z-score normalization
    # normalized_mfccs = (mfccs - mean) / std
    return mfccs


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
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

    # # Assuming cm is your confusion matrix and is a numpy array
    # num_classes = cm.shape[0]
    # class_accuracies = {}
    #
    # for i in range(num_classes):
    #     TP = cm[i, i]
    #     FP = np.sum(cm[:, i]) - TP
    #     FN = np.sum(cm[i, :]) - TP
    #     TN = np.sum(cm) - (FP + FN + TP)
    #
    #     # Calculate the accuracy per class
    #     accuracy = (TP + TN) / (TP + FP + FN + TN)
    #     class_accuracies[classes[i]] = accuracy
    #
    # # Printing accuracies for each class
    # for class_name, accuracy in class_accuracies.items():
    #     print(f"Accuracy for class {class_name}: {accuracy:.2f}")

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


def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27, time_masking_para=100, num_masks=1):
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    warped_mel_spectrogram = tf.identity(mel_spectrogram)

    # Frequency masking
    for i in range(num_masks):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v - f, dtype=tf.int32)
        warped_mel_spectrogram = tf.concat(
            (warped_mel_spectrogram[:f0, :], tf.zeros((f, tau)), warped_mel_spectrogram[f0 + f:, :]), axis=0)

    # Time masking
    for i in range(num_masks):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau - t, dtype=tf.int32)
        warped_mel_spectrogram = tf.concat(
            (warped_mel_spectrogram[:, :t0], tf.zeros((v, t)), warped_mel_spectrogram[:, t0 + t:]), axis=1)

    return warped_mel_spectrogram


def apply_spec_augment(X):
    return np.array([spec_augment(x) for x in X])
