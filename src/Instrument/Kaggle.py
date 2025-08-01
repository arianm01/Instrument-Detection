import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, LSTM, Dense, \
    Activation
from keras.optimizers import Adam
from tcn import TCN
from tensorflow.keras import Sequential, callbacks, layers, regularizers, models
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow import keras

from src.Instrument.Contrastive import create_encoder, add_projection_head, learning_rate, SupervisedContrastiveLoss, \
    create_classifier

# Set constants for the learning rate schedule
INITIAL_LEARNING_RATE = 0.0001
DECAY_RATE = 0.001
MIN_LEARNING_RATE = 0.00005


def lr_time_based_decay(epoch, lr):
    """ Calculate the learning rate based on the initial decay and minimum limit. """
    new_lr = lr * (1 / (1 + DECAY_RATE * epoch))
    return max(new_lr, MIN_LEARNING_RATE)


def cnn_model(input_shape, num_classes, layer_sizes, X_train, y_train, X_test, y_test, instrument, batch_size, epochs):
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
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model_checkpoint_path = f'model_best_CNN_{instrument}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    lr_scheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=batch_size, epochs=epochs,
                     callbacks=[model_checkpoint_callback, lr_scheduler, early_stopping])


def create_advanced_cnn_model(input_shape, num_classes, X_train, y_train, X_test, y_test, instrument):
    """Create an advanced CNN model for Instrument classification."""
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First convolutional block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second convolutional block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third convolutional block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten the output and add dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model_checkpoint_path = f'model_best_CNN_{instrument}.h5'

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[model_checkpoint_callback, early_stopping])

    return model


def cnn_model_binary(input_shape, num_classes, layer_sizes, X_train, y_train, X_test, y_test, instrument, batch_size,
                     epochs, model_checkpoint_path):
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
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    lr_scheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=batch_size, epochs=epochs,
                     callbacks=[model_checkpoint_callback, lr_scheduler, early_stopping])


def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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


def create_classifier_model(input_dim, num_classes, units):
    """ Create a more complex classifier model """
    model = Sequential()

    # Input layer
    model.add(Dense(units[0], input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    for size in units[1:]:
        model.add(Dense(size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_tcn_model(input_shape, num_classes, kernel_size=3, nb_filters=64, dilations=None,
                    dropout_rate=0.3, l1_reg=0.001, l2_reg=0.001):
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]
    model = Sequential([
        TCN(input_shape=input_shape,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout_rate=dropout_rate,
            return_sequences=False),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_contrastive_model(x, y, num_classes):
    """ Train contrastive network using contrastive learning """
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_no, (train_index, test_index) in enumerate(skf.split(x, y_labels), start=1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        batch_size = 4

        model_checkpoint_path = f'model_best_encoder_{fold_no}.keras'
        model_path = f'model_best_classifier_{fold_no}.keras'
        model_checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_path, save_best_only=True,
                                                    monitor='val_loss', mode='min', verbose=1)
        lr_scheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)
        model_callback = ModelCheckpoint(
            filepath=model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        temperature = 0.1
        num_epochs = 100

        # layer_sizes = [512, 256, 128, 64, 32]
        # layer_sizes = [128, 64, 32, 16, 8]
        layer_sizes = [256, 128, 64, 32, 16]

        # encoder = load_model(f'./model_best_encoder_{fold_no}.keras', custom_objects={
        #     'SupervisedContrastiveLoss': SupervisedContrastiveLoss}).layers[1]

        encoder = create_encoder(layer_sizes, input_shape)

        encoder_with_projection_head = add_projection_head(encoder, input_shape)
        encoder_with_projection_head.compile(optimizer=keras.optimizers.Adam(learning_rate),
                                             loss=SupervisedContrastiveLoss(temperature))

        encoder_with_projection_head.summary()

        encoder_with_projection_head.fit(x=x_train, y=y_train, batch_size=batch_size,
                                         validation_data=(x_test, y_test),
                                         epochs=num_epochs,
                                         callbacks=[model_checkpoint_callback, early_stopping, lr_scheduler])

        classifier = create_classifier(encoder, num_classes, input_shape, trainable=False)

        classifier.fit(x=x_train, y=y_tr, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_te),
                       callbacks=[model_callback, early_stopping, lr_scheduler])

        fold_no += 1
