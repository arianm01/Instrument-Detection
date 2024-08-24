import os
from collections import defaultdict

import numpy as np
import pandas as pd
from keras.api.keras import callbacks
from keras.saving.save import load_model
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from src.Infrence.InfrenceInstrument import load_files, extract_label, preprocess_audio
from src.Instrument.Kaggle import lr_time_based_decay, train_contrastive_model
from src.main.main import TIME_FRAME, MERGE_FACTOR, ensemble_learning
from src.utility.utils import get_model_feature

sample_rate = 22050


def load_nava_data():
    dataset_path = '../../../../../archive/NavaDataset'
    files = load_files("../" + dataset_path + '/all_train.txt')
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
    models = [load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_1.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_2.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_3.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_4.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_5.keras')]
    x = split_list_into_arrays(x, models)
    y = y[:x.shape[0]]
    return x, y


def split_list_into_arrays(lst, models, num_chunks=5):
    # Calculate the size of each chunk
    chunk_size = len(lst) // num_chunks

    # Split the list into chunks and convert each chunk to a np.array
    array_chunks = [get_model_feature(np.array(lst[i * chunk_size:(i + 1) * chunk_size])[..., np.newaxis], models) for i
                    in
                    range(num_chunks)]

    return np.concatenate(array_chunks, axis=0)[..., np.newaxis]


def main():
    x, y = load_nava_data()
    models = [load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_1.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_2.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_3.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_4.keras'),
              load_model('../../output/Nava/Contrastive/1 sec/model_best_classifier_5.keras')]
    # histories = train_models(x, y)
    ensemble_learning(x, y, models)
    # histories = train_contrastive_model(x, y, 5)

if __name__ == '__main__':
    main()
