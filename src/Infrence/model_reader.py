import os

from keras.saving.save import load_model

from src.Instrument.Contrastive import SupervisedContrastiveLoss
from src.Instrument.ContrastiveLearning import contrastive_loss


def main():
    path = '../../output/5 class/Contrastive/5 sec/model_best_classifier_2.keras'
    # files = os.listdir(path)

    # for file in files:
    model = load_model(path, custom_objects={'contrastive_loss': contrastive_loss,
                                             'SupervisedContrastiveLoss': SupervisedContrastiveLoss})
    # print(file)
    model.summary()
    model.layers[1].summary()
    print()


# for folder in files:
#     folders = os.listdir('../../../../archive/Instruments/' + folder)
#     for file in folders:
#         print(file)


if __name__ == '__main__':
    main()
