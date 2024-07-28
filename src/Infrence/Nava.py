import os
from collections import defaultdict

import numpy as np
import pandas as pd
from keras.api.keras import callbacks
from keras.saving.save import load_model
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from src.Infrence.InfrenceInstrument import load_files, extract_label, preprocess_audio
from src.Instrument.Kaggle import lr_time_based_decay
from src.main.main import TIME_FRAME, MERGE_FACTOR, ensemble_learning

sample_rate = 22050


def load_nava_data():
    dataset_path = '../../../../../archive/NavaDataset'
    files = load_files("../" + dataset_path)
    true_labels = []
    x, y = [], []

    for file in files:
        print(f"Loading {file}")
        true_label = extract_label(file)
        segments = preprocess_audio(os.path.join("../" + dataset_path, 'Data', file + '.mp3'),
                                    step_size=TIME_FRAME * sample_rate,
                                    segment_duration=TIME_FRAME * MERGE_FACTOR * sample_rate)
        true_labels.extend([true_label] * len(segments))
        x.extend(segments)
        y.extend(true_labels)
        true_labels = []
    y = to_categorical(y)
    x = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels
    return x, y


def tune_models(x, y):
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    print("Tuning models", y_labels.shape, x.shape, y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    model = load_model("../" + model_path)

    for fold_no, (train_index, test_index) in enumerate(skf.split(x, y_labels), start=1):
        print(fold_no, (train_index, test_index))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_checkpoint_path = f'model_best_CNN_{fold_no}.h5'
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        lr_scheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(x_train.shape, y_train.shape)

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                            batch_size=32, epochs=100, callbacks=[model_checkpoint_callback, lr_scheduler])
        histories.append(history)
        fold_no += 1
    return histories


def main():
    x, y = load_nava_data()
    models = [load_model('./model_best_CNN_1.h5'), load_model('./model_best_CNN_2.h5'),
              load_model('./model_best_CNN_3.h5'), load_model('./model_best_CNN_4.h5'),
              load_model('./model_best_CNN_5.h5')]
    # histories = train_models(x, y)
    ensemble_learning(x, y, models)

    # for history in histories:
    #     plot_history(history)


if __name__ == '__main__':
    main()
