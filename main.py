import pandas as pd
import numpy as np
from keras.saving.save import load_model

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from basic_pitch.inference import predict, predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import soundfile as sf
import mido
from midiutil.MidiFile import MIDIFile

from Models.Instrument.Kaggle import CNNModel, lstmModel
from utility import InstrumentDataset
import tensorflow as tf

from utility.InstrumentDataset import plot_history, plot_confusion_matrix
from utility.utils import testGPU

testGPU()


def extract(signal, frame_size, hop_length):
    stft = librosa.stft(signal,
                        n_fft=frame_size,
                        hop_length=hop_length)[:-1]
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram


# signal = load_signal("./input/000001.mp3.wav", 22050)
# temp = extract(signal, 512, 256)
#
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(temp, sr=22050, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram')
# plt.tight_layout()
# plt.show()


x, y, classes = InstrumentDataset.read_data('./input')

# classes = os.listdir('./input')
X = np.array(x)[..., np.newaxis]  # Add an extra dimension for the channels

print(f'the shape of x is {len(x), len(x[0]), len(x[0][0])}')
print(f'the shape of y is {y.shape}')

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#
fold_no = 1
histories = []
for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    his = CNNModel(X_train_fold, y_train_fold, X_test_fold, y_test_fold, fold_no)

    print(f'Training fold {fold_no}...')

    histories.append(his)
    model = load_model(f'model_best_CNN_{fold_no}.h5')
    predictions = model.predict(X_test_fold)
    #
    y_pred_labels = np.argmax(predictions, axis=1)
    #
    # # Ensure y_test is also in the correct format
    # # If y_test is one-hot encoded, convert it to class labels as well
    y_test_labels = np.argmax(y_test_fold, axis=1) if y_test_fold.ndim > 1 else y_test_fold

    plot_confusion_matrix(y_test_labels, y_pred_labels, classes)

    fold_no += 1

# X_train, X_test, y_train, y_test = train_test_split(np.array(x), y, test_size=0.2, random_state=42)
#
# print(X_train.shape, X_test.shape)
#
# his = CNNModel(X_train, y_train, X_test, y_test)
# # his = lstmModel(X_train, y_train, X_test, y_test)
model_path = 'model_best_CNN_1.h5'
#
# # Load the model
#
for history in histories:
    plot_history(history)

# # create your MIDI object
# mf = MIDIFile(1)     # only 1 track
# track = 0   # the only track
#
# time = 0    # start at the beginning
# mf.addTrackName(track, time, "Sample Track")
# mf.addTempo(track, time, 120)
#
# # add some notes
# channel = 0
# volume = 100
#
# pitch = 60           # C4 (middle C)
# time = 0             # start on beat 0
# duration = 1         # 1 beat long
# mf.addNote(track, channel, pitch, time, duration, volume)
#
# pitch = 64           # E4
# time = 2             # start on beat 2
# duration = 1         # 1 beat long
# mf.addNote(track, channel, pitch, time, duration, volume)
#
# pitch = 67           # G4
# time = 4             # start on beat 4
# duration = 1         # 1 beat long
# mf.addNote(track, channel, pitch, time, duration, volume)
#
# # write it to disk
# with open("output.mid", 'wb') as outf:
#     mf.writeFile(outf)
