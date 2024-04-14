import numpy as np
from keras import Sequential
from keras.backend import flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, SeparableConv2D, Add, Normalization, TimeDistributed

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LSTM, \
    Bidirectional

initial_learning_rate = 0.001
decay_rate = 0.001
min_learning_rate = 0.00005  # Set the minimum learning rate limit


def lr_time_based_decay(epoch, lr):
    new_lr = lr * (1 / (1 + decay_rate * epoch))
    if new_lr < min_learning_rate:
        new_lr = min_learning_rate
    return new_lr


def CNNModel(X_train, y_train, X_test, y_test, fold):
    cnn = Sequential()

    # 1 st covolution layer
    cnn.add(Conv2D(128, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    cnn.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))
    # 2 nd covolution layer
    cnn.add(Conv2D(64, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    cnn.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))
    # 3 rd covolution layer
    cnn.add(Conv2D(32, (2, 2), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))

    # ann layer
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    cnn.add(Dropout(0.3))

    model_checkpoint_path = f'model_best_CNN_{fold}.h5'

    # Create a ModelCheckpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)

    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)

    # output
    cnn.add(Dense(y_train.shape[1], activation='softmax'))
    cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return cnn.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=200,
                   callbacks=[model_checkpoint_callback, lr_scheduler])


def lstmModel(X_train, y_train, X_test, y_test, fold):
    model = Sequential()

    # CNN Layer
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # 3 rd covolution layer
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # RNN Layer
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=False))

    # Dense Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model_checkpoint_path = f'model_best_lstm_{fold}.h5'

    # Create a ModelCheckpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)

    # output
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=200,
                     callbacks=[model_checkpoint_callback])
