import numpy as np
from keras.saving.save import load_model
from sklearn.model_selection import StratifiedKFold

from main import evaluate_models
from utility import InstrumentDataset


def train_models(X, y, classes):
    n_splits = 5
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_no, (train_index, test_index) in enumerate(skf.split(X, y_labels), start=1):
        evaluate_models(X[test_index], y[test_index], classes)


def load_data():
    """ Load and preprocess data """
    x, y, classes = InstrumentDataset.read_data('../Dataset', 3)
    X = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels
    print(f'The shape of X is {X.shape}')
    print(f'The shape of y is {y.shape}')
    return X, y, classes


def main():
    X, y, classes = load_data()
    train_models(X, y, classes)


if __name__ == '__main__':
    main()
