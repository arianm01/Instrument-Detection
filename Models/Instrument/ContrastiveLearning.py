import random

import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, LSTM
import tensorflow.keras.backend as K


def create_base_network(input_shape):
    """Base network to be shared (Siamese Network)"""
    _input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(_input)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dropout(0.2)(x)
    return Model(_input, x)


def euclidean_distance(vectors):
    feats_a, feats_b = vectors
    sum_squared = K.sum(K.square(feats_a - feats_b), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = K.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def generate_pairs(X, y):
    """Generate pairs of inputs for contrastive learning."""
    pairs = []
    labels = []

    num_classes = len(np.unique(y))
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    for idx1 in range(len(X)):
        x1 = X[idx1]
        label1 = y[idx1]

        # Positive pair
        idx2 = random.choice(digit_indices[label1])
        x2 = X[idx2]
        pairs.append([x1, x2])
        labels.append(1)

        # Negative pair
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)
        idx2 = random.choice(digit_indices[label2])
        x2 = X[idx2]
        pairs.append([x1, x2])
        labels.append(0)

    return np.array(pairs), np.array(labels)


def generate_embeddings(model, x):
    """ Generate embeddings for the given data using the trained model """
    embedding_model = Model(inputs=model.input[0], outputs=model.layers[2].output)
    embeddings = embedding_model.predict([x, x])
    return embeddings
