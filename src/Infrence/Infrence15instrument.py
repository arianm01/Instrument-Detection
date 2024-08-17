import numpy as np

from src.Infrence.InfrenceInstrument import predict_segments, load_models, evaluate_predictions
from src.main.main import MERGE_FACTOR, TIME_FRAME
from src.utility import InstrumentDataset


def get_predicted_results(model, models, model_base, x):
    return predict_segments(x, model, models, model_base, False)


def main():
    audio_path = '../../Dataset'

    x, y, _ = InstrumentDataset.read_data(audio_path, MERGE_FACTOR, TIME_FRAME,
                                          folder='../../Models/Instrument/splits/test', balance_needed=False)
    models_addr = ['../../output/5 class/CNN/5 seconds/model_best_CNN_1.h5',
                   '../../output/5 class/CNN/5 seconds/model_best_CNN_5.h5',
                   '../../output/5 class/CNN/5 seconds/model_best_CNN_6.h5',
                   '../../output/5 class/CNN/5 seconds/model_best_CNN_8.h5',
                   '../../output/5 class/CNN/5 seconds/model_best_CNN_9.h5']
    model, models, model_base = load_models('../../output/5 class/CNN/10 seconds/ensemble.keras', models_addr,
                                            models_addr[0])

    x = np.array(x)
    x = x[..., np.newaxis]
    predicted_labels = get_predicted_results(model, models, model_base, x)
    y = np.argmax(y, axis=1)
    evaluate_predictions(y, predicted_labels)


if __name__ == "__main__":
    main()
