import os

import numpy as np

from Models.Instrument.InfrenceInstrument import load_models, evaluate_predictions, predict_segments
from main import MERGE_FACTOR, TIME_FRAME
from utility import InstrumentDataset


def get_predicted_results(model, models, model_base, x):
    return predict_segments(x, model, models, model_base, False)


def main():
    audio_path = '../audio_segments_test'
    x, y, _ = InstrumentDataset.read_data(audio_path, MERGE_FACTOR, TIME_FRAME, folder='test', balance_needed=False)
    model, models, model_base = load_models('../../../model_best_CNN_2.h5', [], '')

    x = np.array(x)
    x = x[..., np.newaxis]
    predicted_labels = get_predicted_results(model, models, model_base, x)
    y = np.argmax(y, axis=1)
    evaluate_predictions(y, predicted_labels)


if __name__ == "__main__":
    main()
