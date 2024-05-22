import os

import keras
import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda
from keras.models import load_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
from sklearn.utils import compute_class_weight

from Models.Instrument.ContrastiveLearning import generate_pairs, create_base_network, euclidean_distance, \
    contrastive_loss, generate_embeddings
from Models.Instrument.Kaggle import cnn_model, custom_model
from utility import InstrumentDataset
from utility.EuclideanDistanceLayer import EuclideanDistanceLayer
from utility.InstrumentDataset import plot_history, plot_confusion_matrix
from utility.utils import test_gpu, reshape_data

TIME_FRAME = 5

# Initialize GPU configuration
test_gpu()


def extract_features(signal, frame_size, hop_length):
    """ Extract log spectrogram features from the signal """
    stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)[:-1]
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram


def load_data():
    """ Load and preprocess data """
    x, y, classes = InstrumentDataset.read_data('Models/Instrument/audio_segments_test', 1, duration=1)
    print(np.array(x).shape)
    X = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels
    print(f'The shape of X is {X.shape}')
    print(f'The shape of y is {y.shape}')
    return X, y, classes


def train_models(x, y):
    """ Train models using K-fold cross-validation """
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    # X = reshape_data(X, TIME_FRAME)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []

    for fold_no, (train_index, test_index) in enumerate(skf.split(x, y_labels), start=1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f'Training fold {fold_no}...')
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        num_classes = y_train.shape[1]
        layer_sizes = [256, 128, 64, 32]
        # history = custom_model(input_shape, num_classes, fold_no, X_train, y_train, X_test, y_test)
        history = cnn_model(input_shape, num_classes, layer_sizes, x_train, y_train, x_test, y_test, fold_no, 256, 200)
        histories.append(history)
        fold_no += 1

    return histories


def evaluate_models(x, y, classes):
    """ Evaluate models on a separate test set """
    model_path = '../Models/Instrument/Finetune/model_best_CNN_1.h5'
    model = load_model(model_path)
    predictions = model.predict(x)
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
    plot_confusion_matrix(y_true_labels, y_pred_labels, classes)


def train_contrastive_model(x, y):
    """ Train Siamese network using contrastive learning """
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []

    for fold_no, (train_index, test_index) in enumerate(skf.split(x, y_labels), start=1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]

        pairs_train, labels_train = generate_pairs(x_train, y_train)
        pairs_test, labels_test = generate_pairs(x_test, y_test)

        input_shape = x_train.shape[1:]
        base_network = create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        feat_a = base_network(input_a)
        feat_b = base_network(input_b)

        distance = EuclideanDistanceLayer()([feat_a, feat_b])

        model = Model(inputs=[input_a, input_b], outputs=distance)

        model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])

        model_checkpoint_path = f'model_best_Siamese_{fold_no}.keras'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

        # y_t = y.ravel()  # Flatten the array to 1D
        # class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y_t)
        # class_weights_dict = dict(enumerate(class_weights))

        history = model.fit(
            [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
            validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test),
            batch_size=128,
            epochs=200,
            callbacks=[model_checkpoint_callback],
            # class_weight=class_weights_dict
        )

        histories.append(history)
        fold_no += 1

    return histories


def evaluate_contrastive_model(x, y, classes):
    """ Evaluate Siamese network on a separate test set """
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    model_path = 'model_best_Siamese_1.keras'  # Adjust the path as necessary
    model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss,
                                                   'EuclideanDistanceLayer': EuclideanDistanceLayer})

    # embeddings = generate_embeddings(model, x)

    pairs_test, labels_test = generate_pairs(x, y_labels)
    predictions = model.predict([pairs_test[:, 0], pairs_test[:, 1]])
    y_pred = (predictions < 0.5).astype(int).flatten()

    y_true = labels_test

    # Debugging output
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    i, j = 0, 0
    for k in range(56000):
        if y_true[k] == y_pred[k]:
            i += 1
        if y_pred[k] == 0:
            j += 1
    print("i:", i / 560, j)


# plot_confusion_matrix(y_true, y_pred, classes)


def main():
    model_path = 'model_best_Siamese_1.keras'  # Adjust the path as necessary
    model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss,
                                                   'EuclideanDistanceLayer': EuclideanDistanceLayer})
    model.summary()
    x, y, classes = load_data()
    # histories = train_models(x, y)
    histories = train_contrastive_model(x, y)

    # for history in histories:
    #     plot_history(history)
    # evaluate_contrastive_model(x, y, classes)


if __name__ == '__main__':
    main()
