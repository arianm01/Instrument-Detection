import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import librosa

from utility.utils import balance_dataset_with_augmentation, create_label_mapping, convert_labels_to_indices


def load_signal(file_path, SAMPLE_RATE):
    signal = librosa.load(file_path,
                          sr=SAMPLE_RATE)[0]
    return signal


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
    x, y, signals = [], [], []
    classes = os.listdir(dataset_path)
    print(classes)
    for instrument in classes:
        print(instrument)
        files = os.listdir(os.path.join(dataset_path, str(instrument)))
        baseSignal = None
        for i, file in enumerate(tqdm(files)):
            file_path = os.path.join(dataset_path, str(instrument), str(file))
            signal, sample_rate = librosa.load(file_path, duration=duration, sr=16000)
            if i == 3600:
                break
            seg = i + 1
            if seg % merge_factor != 0:
                if baseSignal is not None:
                    baseSignal = np.concatenate([baseSignal, signal], axis=0)
                else:
                    baseSignal = signal
                continue
            else:
                if baseSignal is not None:
                    baseSignal = np.concatenate([baseSignal, signal], axis=0)
                else:
                    baseSignal = signal
            MFCCs = librosa.feature.mfcc(y=baseSignal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                         n_mfcc=n_mfcc)
            # here we will use both of these values and see which one is better
            # I think the use of the transpose is for the shapes to be correct
            mfcc = MFCCs.T
            x.append(mfcc)
            y.append(instrument)
            baseSignal = None

    max_length = max(mfcc.shape[0] for mfcc in x)
    x_padded = [np.pad(mfcc, ((0, max_length - mfcc.shape[0]), (0, 0)), mode='constant') if mfcc.shape[
                                                                                                0] < max_length else mfcc[
                                                                                                                     :max_length]
                for mfcc in x]

    # Flatten MFCC features for SMOTE
    x_flat = [mfcc.flatten() for mfcc in x_padded]
    print(len(x), len(y))

    label_to_index, index_to_label = create_label_mapping(y)
    y = convert_labels_to_indices(y, label_to_index)

    target_count = max(np.bincount(y))  # Adjust target count as needed
    print(target_count)
    x = np.array(x)
    x, y = balance_dataset_with_augmentation(x, y, 16000, target_count)
    y = np.array(pd.get_dummies(y))
    # Resample using SMOTE
    # x_resampled, y_resampled = SMOTEENN().fit_resample(np.array(x_flat), y)
    #
    # # Reshape resampled data back to original MFCC shape
    # mfcc_shape = x_padded[0].shape
    # x_resampled = [resampled.reshape(mfcc_shape) for resampled in x_resampled]

    print(len(x), len(y))
    return x, y, classes


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
