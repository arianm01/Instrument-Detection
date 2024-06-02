import numpy as np
import librosa
from keras.layers import TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, LSTM, Dense, \
    Activation
from sklearn.utils import class_weight
from tensorflow.keras import Sequential, callbacks, layers, regularizers, models
from sklearn.model_selection import KFold

# Set constants for the learning rate schedule
INITIAL_LEARNING_RATE = 0.0001
DECAY_RATE = 0.0001
MIN_LEARNING_RATE = 0.00005


def lr_time_based_decay(epoch, lr):
    """ Calculate the learning rate based on the initial decay and minimum limit. """
    new_lr = lr * (1 / (1 + DECAY_RATE * epoch))
    return max(new_lr, MIN_LEARNING_RATE)


def cnn_model(input_shape, num_classes, layer_sizes, X_train, y_train, X_test, y_test, fold, batch_size, epochs):
    """ Build and train a CNN model with specified architecture and hyperparameters. """
    model = Sequential()
    model.add(layers.Conv2D(layer_sizes[0], (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    for size in layer_sizes[1:]:
        model.add(layers.Conv2D(size, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model_checkpoint_path = f'model_best_CNN_{fold}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    lr_scheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback, lr_scheduler])


def build_lstm_model(input_shape, num_classes, fold, X_train, y_train, X_test, y_test):
    """ Build and train an LSTM model for sequence processing. """
    model = Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_checkpoint_path = f'model_best_lstm_{fold}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=256, epochs=200, callbacks=[model_checkpoint_callback])


def custom_model(input_shape, num_classes, fold, X_train, y_train, X_test, y_test):
    model = Sequential()
    # First Conv Block
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    # Second Conv Block
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    # LSTM Layer
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128))

    # Dense Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_checkpoint_path = f'model_best_lstm_{fold}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=256, epochs=200, callbacks=[model_checkpoint_callback])


def create_classifier_model(input_dim, num_classes):
    """ Create a more complex classifier model """
    model = Sequential()

    # Input layer
    model.add(Dense(512, input_shape=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Hidden layer 1
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Hidden layer 2
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
