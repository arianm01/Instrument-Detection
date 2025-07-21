import random
from datetime import datetime

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, LSTM
import tensorflow.keras.backend as K
from keras.optimizers import Adam
from keras.saving.save import load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def create_base_network(input_shape):
    """Base network to be shared (Siamese Network)"""
    _input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(_input)
    x = MaxPooling2D(padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(padding='same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dropout(0.3)(x)
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

        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)
        idx2 = random.choice(digit_indices[label2])
        x2 = X[idx2]
        pairs.append([x1, x2])
        labels.append(0)

    return np.array(pairs), np.array(labels)


def generate_embeddings(model, x, model_name):
    base_network = model.get_layer(model_name)
    embedding_model = Model(inputs=base_network.input, outputs=base_network.output)
    embeddings = embedding_model.predict(x)
    return embeddings


def create_classifier_on_siamese(base_model, input_shape, num_classes):
    for layer in base_model.layers:
        layer.trainable = True  # Set all layers to trainable

    model_input = Input(shape=input_shape)
    x = base_model(model_input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    classifier_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=classifier_output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model
