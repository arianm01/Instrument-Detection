import numpy as np
import librosa
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, LSTM, Dense, \
    Activation
from keras.optimizers import Adam
from keras.regularizers import l2, l1, l1_l2
from keras.saving.save import load_model
from sklearn.utils import class_weight
from tcn import TCN
from tensorflow.keras import Sequential, callbacks, layers, regularizers, models
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from Models.Instrument.ContrastiveLearning import generate_pairs, create_base_network, contrastive_loss, \
    generate_embeddings
from utility.EuclideanDistanceLayer import EuclideanDistanceLayer

# Set constants for the learning rate schedule
INITIAL_LEARNING_RATE = 0.0001
DECAY_RATE = 0.0005
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
    model.add(layers.Dense(num_classes, activation='softmax'))

    model_checkpoint_path = f'model_best_CNN_{instrument}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    lr_scheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=batch_size, epochs=epochs,
                     callbacks=[model_checkpoint_callback, lr_scheduler, early_stopping])


def create_advanced_cnn_model(input_shape, num_classes, X_train, y_train, X_test, y_test, instrument):
    """Create an advanced CNN model for instrument classification."""
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

        model.compile(loss=contrastive_loss, optimizer='adam')

        model_checkpoint_path = f'model_best_Siamese_{fold_no}.keras'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

        # y_t = y.ravel()  # Flatten the array to 1D
        # class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y_t)
        # class_weights_dict = dict(enumerate(class_weights))

        history = model.fit(
            [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
            validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test),
            batch_size=64,
            epochs=200,
            callbacks=[model_checkpoint_callback],
            # class_weight=class_weights_dict
        )

        histories.append(history)
        fold_no += 1

    return histories


def evaluate_contrastive_model(x, y, classes):
    """ Evaluate Siamese network on a separate test set """
    model_path = 'model_best_Siamese_1.keras'  # Adjust the path as necessary
    model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss,
                                                   'EuclideanDistanceLayer': EuclideanDistanceLayer})

    embeddings = generate_embeddings(model, x, 'model')

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    # Define classifier model

    # Train classifier
    input_shape = (embeddings.shape[1],)
    num_classes = y.shape[1]
    model_checkpoint_path = 'model_best_contrastive_1.keras'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    classifier_model = create_classifier_model(input_shape, num_classes)
    classifier_model.summary()
    history = classifier_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=32,
                                   callbacks=[model_checkpoint_callback])
