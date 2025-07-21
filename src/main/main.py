import librosa
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.Instrument.Kaggle import (
    cnn_model, lr_time_based_decay, create_classifier_model, train_contrastive_model
)
from src.utility import InstrumentDataset
from src.utility.InstrumentDataset import plot_confusion_matrix, \
    extract_and_visualize_features
from src.utility.utils import test_gpu, get_model_feature

TIME_FRAME = 1
MERGE_FACTOR = 1


def extract_features(signal, frame_size, hop_length):
    """Extract log spectrogram features from the signal."""
    stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)[:-1]
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram


def load_data(split):
    """Load and preprocess data."""
    x, y, classes = InstrumentDataset.read_data(
        '../../Dataset', MERGE_FACTOR, TIME_FRAME, folder=f'../../Models/split/{split}'
    )
    print(np.array(x).shape)
    X = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels
    print(f'The shape of X is {X.shape}')
    print(f'The shape of y is {y.shape}')
    return X, y, classes


def train_models(x, y):
    """Train models using K-fold cross-validation."""
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []

    for fold_no, (train_index, test_index) in enumerate(skf.split(x, y_labels), start=1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if fold_no < 2:
            continue

        print(f'Training fold {fold_no}...')
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        num_classes = y_train.shape[1]
        layer_sizes = [256, 128, 64, 32, 16]
        history = cnn_model(input_shape, num_classes, layer_sizes, x_train, y_train, x_test, y_test, fold_no, 16, 100)
        histories.append(history)

    return histories


def evaluate_models(x, y, classes):
    """Evaluate models on a separate test set."""
    model_path = '../../Models/splits/1/model_best_classifier_1.keras'
    model = load_model(model_path)
    predictions = model.predict(x)
    instrument_names = [
        'Daf', 'Divan', 'Dutar', 'Gheychak', 'Kamancheh',
        'Ney Anban', 'Ney', 'Oud', 'Qanun', 'Rubab',
        'Santur', 'Setar', 'Tanbour', 'Tar', 'Tonbak'
    ]
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
    plot_confusion_matrix(y_true_labels, y_pred_labels, instrument_names)
    # Extract features using your encoder architecture
    embeddings_2d, features = extract_and_visualize_features(
        model, x, y, instrument_names, n_samples=10000
    )


def train_meta_model(X_train, y_train, x_test, y_test):
    """Train meta-model using meta-features."""
    input_shape = X_train.shape[1]
    model = create_classifier_model(input_shape, 7, [4096, 2048, 1024, 512, 256, 128])
    model_checkpoint_callback = ModelCheckpoint(filepath='ensemble.keras', save_best_only=True, monitor='val_loss',
                                                mode='min', verbose=1)
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test),
              callbacks=[model_checkpoint_callback, lr_scheduler, early_stopping])
    return model


def ensemble_learning(x, y, models):
    """Perform ensemble learning using meta-features."""
    meta_features = get_model_feature(x, models)
    X_train, X_val, y_train, y_val = train_test_split(meta_features, y, test_size=0.1, random_state=42)
    train_meta_model(X_train, y_train, X_val, y_val)


def main():
    test_gpu()
    x, y, classes = load_data('test')
    train_contrastive_model(x, y, len(classes))
    evaluate_models(x, y, classes)


if __name__ == '__main__':
    main()
