import os

import numpy as np
import pandas as pd
from keras.saving.save import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
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
              folder='./Models/Instrument/splits/train',
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
    classes = ['Tar', 'Kamancheh', 'Santur', 'Setar', 'Ney']
    # classes = os.listdir(dataset_path)
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
        models = [load_model('../../output/5 classes/Contrastive/1 sec/model_best_classifier_1.keras'),
                  load_model('../../output/5 classes/Contrastive/1 sec/model_best_classifier_2.keras'),
                  load_model('../../output/5 classes/Contrastive/1 sec/model_best_classifier_3.keras'),
                  load_model('../../output/5 classes/Contrastive/1 sec/model_best_classifier_4.keras'),
                  load_model('../../output/5 classes/Contrastive/1 sec/model_best_classifier_5.keras')]
        x, y = balance_dataset_with_augmentation(x, y, 22050, target_count, models)
    y = np.array(pd.get_dummies(y))
    return x, y, classes


def process_files(files, dataset_path, instrument, merge_factor, duration, n_mfcc, n_fft, hop_length, x, y, label,
                  window_size, step_size):
    base_signal, seg, last_file = [], 1, ''

    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(dataset_path, instrument, file)
        signal, sample_rate = librosa.load(file_path, duration=duration)

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
