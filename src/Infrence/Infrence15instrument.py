import numpy as np

from src.Infrence.InfrenceInstrument import predict_segments, load_models, evaluate_predictions
from src.main.main import TIME_FRAME
from src.utility import InstrumentDataset


def get_predicted_results(model, models, model_base, x):
    return predict_segments(x, model, models, model_base, False)


def main():
    audio_path = '../../Dataset'

    x, y, _ = InstrumentDataset.read_data(audio_path, 2, TIME_FRAME,
                                          folder='../../Models/split/test', balance_needed=False)
    models_addr = ['../../src/main/model_best_classifier_1.keras',
                   '../../output/15 classes/1 sec/1 sec/model_best_classifier_2.keras',
                   '../../output/15 classes/1 sec/1 sec/model_best_classifier_3.keras',
                   '../../output/15 classes/1 sec/1 sec/model_best_classifier_4.keras',
                   '../../output/15 classes/1 sec/1 sec/model_best_classifier_5.keras']
    path = '../../output/15 classes/Contrastive/10 sec/ensemble.keras'
    model, models, model_base = load_models(path, models_addr, models_addr[0])

    x = np.array(x)
    x = x[..., np.newaxis]
    predicted_labels = get_predicted_results(model, models, model_base, x)
    y = np.argmax(y, axis=1)
    evaluate_predictions(y, predicted_labels, True)


if __name__ == "__main__":
    main()
