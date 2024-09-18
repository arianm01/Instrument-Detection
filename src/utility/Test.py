import os

import librosa
import numpy as np
import pandas as pd
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from src.main.main import evaluate_models
from src.utility import InstrumentDataset

DATASET_PATH = '../Dataset'

def train_models(x, y, classes):
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_no, (train_index, test_index) in enumerate(skf.split(x, y_labels), start=1):
        evaluate_models(x[test_index], y[test_index], classes)


def load_data():
    """ Load and preprocess data """
    x, y, classes = InstrumentDataset.read_data(DATASET_PATH, 3)
    X = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels
    print(f'The shape of X is {X.shape}')
    print(f'The shape of y is {y.shape}')
    return X, y, classes


def show_class_distribution():
    df = pd.read_csv('../output/Instruments.csv', delimiter=',')

    _, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(df['count'], labels=df['Instruments'], autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()


def cal_fft(signal, sample_rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1 / sample_rate)
    Y = abs(np.fft.rfft(signal) / n)
    return Y, freq


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=8, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(8):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=8, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(8):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=8, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(8):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=8, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(8):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def main():
    X, y, classes = load_data()
    # train_models(X, y, classes)
    # showClassDistribution()
    represent_data()


def represent_data():
    signals, ffts, fbanks, mfccs = {}, {}, {}, {}
    instruments = os.listdir(DATASET_PATH)
    for instrument in instruments:
        instrument_path = os.path.join(DATASET_PATH, str(instrument))
        file = os.listdir(instrument_path)[0]
        file_path = os.path.join(instrument_path, str(file))
        signal, sample_rate = librosa.load(file_path)
        print(sample_rate)
        mask = envelope(signal, sample_rate, 0.0005)
        signal = signal[mask]
        signals[instrument] = signal
        ffts[instrument] = cal_fft(signal, sample_rate)

        S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=26, n_fft=2048, hop_length=512)
        fbanks[instrument] = librosa.power_to_db(S, ref=np.max).T
        mel = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=2048, hop_length=512, n_mfcc=26)
        mfccs[instrument] = mel
    plot_signals(signals)
    plt.show()
    plot_fft(ffts)
    plt.show()
    plot_mfccs(mfccs)
    plt.show()
    plot_fbank(fbanks)
    plt.show()


if __name__ == '__main__':
    main()
