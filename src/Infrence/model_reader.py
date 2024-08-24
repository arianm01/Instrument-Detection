import os

import visualkeras
from keras.utils import plot_model

from keras.saving.save import load_model
import tensorflow as tf
from src.Instrument.Contrastive import SupervisedContrastiveLoss
from src.Instrument.ContrastiveLearning import contrastive_loss
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = '../../output/Nava/Contrastive/1 sec/model_best_encoder_1.keras'
    # model = load_model(path)
    # plot_model(model, to_file='./model.png', show_shapes=True)
    #
    # # for file in files:
    model = load_model(path, custom_objects={'contrastive_loss': contrastive_loss,
                                             'SupervisedContrastiveLoss': SupervisedContrastiveLoss})
    # # print(file)
    model.summary()
    model.layers[1].summary()
    # print()
    # for folder in files:
    #     folders = os.listdir('../../../../archive/Instruments/' + folder)
    #     for file in folders:
    #         print(file)


if __name__ == '__main__':
    main()
