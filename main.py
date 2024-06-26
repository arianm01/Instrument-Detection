import os

import numpy as np
import librosa
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.utils.version_utils import callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from tcn import tcn_full_summary

from Models.Instrument.Kaggle import cnn_model, create_classifier_model, build_tcn_model, \
    lr_time_based_decay, cnn_model_binary
from Models.MixtureExperts import train, create_gating_network, generate_performance_labels
from Models.TransformerModel import build_transformer_model
from utility import InstrumentDataset
from utility.InstrumentDataset import plot_confusion_matrix, separate_and_balance_data, get_meta_features
from utility.utils import test_gpu, sanitize_file_name

TIME_FRAME = 1
MERGE_FACTOR = 10

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
    x, y, classes = InstrumentDataset.read_data('./Models/Instrument/audio_segments_test', MERGE_FACTOR, TIME_FRAME)
    # x, y, classes = InstrumentDataset.read_data('./Dataset', MERGE_FACTOR, TIME_FRAME)
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

        if fold_no < 5:
            continue

        print(f'Training fold {fold_no}...')
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        num_classes = y_train.shape[1]
        layer_sizes = [512, 256, 128, 64]
        # history = custom_model(input_shape, num_classes, fold_no, X_train, y_train, X_test, y_test)
        history = cnn_model(input_shape, num_classes, layer_sizes, x_train, y_train, x_test, y_test, fold_no, 8, 100)
        # history = create_advanced_cnn_model(input_shape, num_classes, x_train, y_train, x_test, y_test, fold_no)
        # history = tcn_model(num_classes, x_test, x_train, y_test, y_train)
        # history = transformer(input_shape, num_classes, x_test, x_train, y_test, y_train)
        histories.append(history)
        fold_no += 1

    return histories


def transformer(input_shape, num_classes, x_test, x_train, y_test, y_train):
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_blocks = 2
    model = build_transformer_model(input_shape, num_classes, embed_dim, num_heads, ff_dim, num_blocks)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint_callback = ModelCheckpoint(filepath='transformer.keras', save_best_only=True, monitor='val_loss',
                                                mode='min', verbose=1)
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32,
                        callbacks=[model_checkpoint_callback, lr_scheduler])
    return history


def tcn_model(num_classes, x_test, x_train, y_test, y_train):
    input_shape = (x_train.shape[1], x_train.shape[2])
    # Build the TCN model
    tcn_model = build_tcn_model(input_shape, num_classes)
    # Print the model summary
    tcn_full_summary(tcn_model)
    # Define the model checkpoint and learning rate scheduler
    model_checkpoint_callback = ModelCheckpoint(filepath='tcn_model.h5', save_best_only=True, monitor='val_loss',
                                                mode='min', verbose=1)
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    # Train the TCN model
    history = tcn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32,
                            callbacks=[model_checkpoint_callback, lr_scheduler])
    # Save the model
    tcn_model.save('tcn_model.h5')
    return history


def evaluate_models(x, y, classes):
    """ Evaluate models on a separate test set """
    model_path = '../Models/Instrument/Finetune/model_best_CNN_1.h5'
    model = load_model(model_path)
    predictions = model.predict(x)
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
    plot_confusion_matrix(y_true_labels, y_pred_labels, classes)


def train_meta_model(X_train, y_train, x_test, y_test, models):
    """Train meta-model using meta-features."""
    input_shape = X_train.shape[1]

    model = create_classifier_model(input_shape, 5, [64, 32, 16])
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath='ensemble.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test),
              callbacks=[model_checkpoint_callback])
    return model


def train_models_by_instrument(X, y, instruments, save_dir="models"):
    """Train a separate binary classifier model for each instrument."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    instrument_data = separate_and_balance_data(X, y, instruments)
    instrument_models = {}

    for instrument, (X_inst, y_inst) in instrument_data.items():
        print("Training model for instrument", X_inst.shape, instrument, y_inst.shape)

        # skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=42)
        # histories = []
        #
        # for fold_no, (train_index, test_index) in enumerate(skf.split(X_inst, y_labels), start=1):
        #     x_train, x_test = X_inst[train_index], X_inst[test_index]
        #     y_train, y_test = y_inst[train_index], y_inst[test_index]
        x_train, x_test, y_train, y_test = train_test_split(X_inst, y_inst, test_size=0.2, random_state=42)

        # Create a sanitized file name for saving the model
        model_file_name = sanitize_file_name(f"model_best_{instrument}.h5")
        model_file_path = os.path.join(save_dir, model_file_name)
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        num_classes = 1
        layer_sizes = [128, 64]
        print(f"Training model for instrument: {instrument}")
        history = cnn_model_binary(input_shape, num_classes, layer_sizes, X_inst, y_inst, x_test, y_test,
                                   instrument, 128, 50, model_file_path)

    return instrument_models


def ensemble_learning(x, y, instruments):
    models = [load_model('./model_best_CNN_1.h5'), load_model('./model_best_CNN_6.h5'),
              load_model('./model_best_CNN_7.h5'), load_model('./model_best_CNN_4.h5'),
              load_model('./model_best_CNN_5.h5')]
    meta_features = get_model_feature(x, models)
    # instrument_models = train_models_by_instrument(x, y, instruments)

    # y_meta_train = np.argmax(y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(meta_features, y, test_size=0.2, random_state=42)

    meta_model = train_meta_model(X_train, y_train, X_val, y_val, models)


def get_model_feature(x, models=None):
    if models is None:
        model_paths = ['./Models/model_best_Tar.h5', './Models/model_best_Kamancheh.h5',
                       './Models/model_best_Santur.h5', './Models/model_best_Setar.h5',
                       './Models/model_best_Ney.h5']
        print(models)
        model1 = load_model(model_paths[0])
        model2 = load_model(model_paths[1])
        model3 = load_model(model_paths[2])
        model4 = load_model(model_paths[3])
        model5 = load_model(model_paths[4])
        models = [model1, model2, model3, model4, model5]
    meta_features = get_meta_features(models, x)
    return meta_features


def expert_training(x, y, classes):
    models = [load_model('./model_best_CNN_1.h5'), load_model('./model_best_CNN_4.h5'),
              load_model('./model_best_CNN_5.h5'), load_model('./model_best_CNN_6.h5'),
              load_model('./model_best_CNN_7.h5')]

    # Generate labels for the first-level gating network
    first_level_labels = generate_performance_labels(models, x, y)

    print(first_level_labels)

    X_train, X_val, y_train, y_val = train_test_split(x, first_level_labels, test_size=0.2, random_state=42)
    train(X_train, y_train, X_val, y_val, models)


def main():
    x, y, classes = load_data()
    # histories = train_models(x, y)
    # histories = train_contrastive_model(x, y)
    ensemble_learning(x, y, classes)
    # expert_training(x, y, classes)

    # for history in histories:
    #     plot_history(history)
    # evaluate_contrastive_model(x, y, classes)
    # evaluate_combined_contrastive_model(x, y, classes)


if __name__ == '__main__':
    main()
