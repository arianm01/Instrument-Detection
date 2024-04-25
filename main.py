import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from Models.Instrument.Kaggle import cnn_model, custom_model
from utility import InstrumentDataset
from utility.InstrumentDataset import plot_history, plot_confusion_matrix
from utility.utils import testGPU, reshape_data

TIME_FRAME = 5

# Initialize GPU configuration
testGPU()


def extract_features(signal, frame_size, hop_length):
    """ Extract log spectrogram features from the signal """
    stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)[:-1]
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram


def load_data():
    """ Load and preprocess data """
    x, y, classes = InstrumentDataset.read_data('Dataset', 1)
    X = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels
    print(f'The shape of X is {X.shape}')
    print(f'The shape of y is {y.shape}')
    return X, y, classes


def train_models(X, y, classes):
    """ Train models using K-fold cross-validation """
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    X = reshape_data(X, TIME_FRAME)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []

    for fold_no, (train_index, test_index) in enumerate(skf.split(X, y_labels), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f'Training fold {fold_no}...')
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
        num_classes = y_train.shape[1]
        layer_sizes = [512, 256, 128, 64, 32]
        history = custom_model(input_shape, num_classes, fold_no, X_train, y_train, X_test, y_test)
        histories.append(history)
        fold_no += 1

    return histories


def evaluate_models(X, y, classes):
    """ Evaluate models on a separate test set """
    model_path = '../Models/Instrument/Finetune/model_best_CNN_1.h5'
    model = load_model(model_path)
    predictions = model.predict(X)
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
    plot_confusion_matrix(y_true_labels, y_pred_labels, classes)


def main():
    X, y, classes = load_data()
    histories = train_models(X, y, classes)

    for history in histories:
        plot_history(history)


if __name__ == '__main__':
    main()
