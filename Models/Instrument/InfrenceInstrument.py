import os
import librosa
import numpy as np
from collections import Counter

import pandas as pd
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.models import load_model

from Models.Instrument.ContrastiveLearning import generate_embeddings, contrastive_loss
from utility.EuclideanDistanceLayer import EuclideanDistanceLayer
from utility.InstrumentDataset import plot_confusion_matrix, compute_mfcc

# label_mapping = {0: 4, 1: 0, 2: 2, 3: 3, 4: 1}


def preprocess_audio(audio_path, segment_duration=5, n_mfcc=13):
    try:
        signal, sample_rate = librosa.load(audio_path, sr=16000)
        samples_per_segment = int(sample_rate * segment_duration)
        num_segments = int(np.ceil(len(signal) / samples_per_segment))
        segments = []

        for segment in range(num_segments):
            start_sample = samples_per_segment * segment
            end_sample = start_sample + samples_per_segment
            if end_sample <= len(signal):
                segment_signal = signal[start_sample:end_sample]
                mfcc = compute_mfcc(segment_signal, sample_rate, n_mfcc, 512, 512)
                segments.append(mfcc)
        return np.array(segments)
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return np.array([])


def predict_contrastive(segments):
    embeddings = generate_embeddings(model_base, segments, 'model')
    return model.predict(embeddings)


def predict(audio_path, model):
    segments = preprocess_audio(audio_path)
    if segments.size == 0:
        return np.array([])
    segments = segments[..., np.newaxis]
    try:
        if contrastive:
            predictions = predict_contrastive(segments)
        else:
            predictions = model.predict(segments)
        return np.argmax(predictions, axis=1)
    except Exception as e:
        print(f"Error during prediction for {audio_path}: {str(e)}")
        return np.array([])


audio_path = '../../../../archive/NavaDataset'
try:
    with open(audio_path + '/dev.txt', 'r') as file:
        files = [line.strip() for line in file.readlines()]
except FileNotFoundError:
    print("File not found.")
    files = []

# classes = os.listdir('../../Dataset')
classes = ['Tar', 'Kamancheh', 'Santur', 'Setar', 'Ney']
true_labels = []
predicted_labels = []


# def extract_label(file_name):
#     try:
#         return label_mapping[int(file_name[0])]
#     except (IndexError, ValueError, KeyError):
#         return None


model = load_model('../../model_best_CNN_1.h5')
model_base = load_model('../../model_best_Siamese_1.keras',
                        custom_objects={'contrastive_loss': contrastive_loss,
                                        'EuclideanDistanceLayer': EuclideanDistanceLayer})
contrastive = False

for file in files:
    # true_label = extract_label(file)
    true_label = file[0]
    predicted_label = predict(audio_path + '/Data/' + file + '.mp3', model)
    true_labels.extend([true_label] * len(predicted_label))
    predicted_labels.extend(predicted_label)
    if predicted_label.size > 0:
        label_counts = Counter(predicted_label)
        for label, count in label_counts.items():
            class_name = classes[label]
            print(f"Class '{class_name}' appears {count} times in predictions for {file}.")
    else:
        print(f"No predictions for {file}.")

# plot_confusion_matrix(true_labels, predicted_labels, list(label_mapping.values()))
print(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print('Confusion matrix', conf_matrix)
class_accuracies = {}
total_true_positives, total_predict = 0, 0
for i, class_label in enumerate(label_mapping.values()):
    true_positives = conf_matrix[i, i]
    total_predictions = conf_matrix[:, i].sum()
    total_true_positives += true_positives
    total_predict += total_predictions
    if total_predictions > 0:
        accuracy = true_positives / total_predictions
    else:
        accuracy = 0  # To handle division by zero if there are no predictions for this class
    class_accuracies[i] = accuracy

# Print accuracy for each class
for label, acc in class_accuracies.items():
    print(f"Accuracy for class {label}: {acc * 100:.2f}")

accuracy = total_true_positives / total_predict
print(f"Total Accuracy {total_true_positives}, {total_predict}: {accuracy * 100}")

f1_scores = f1_score(true_labels, predicted_labels, average=None)
macro_f1_score = f1_score(true_labels, predicted_labels, average='weighted')

# Print F1 score for each class
for i, label in enumerate(label_mapping.values()):
    print(f"F1 Score for class {label}: {f1_scores[i]:.2f}")

print(f"Macro-average F1 Score: {macro_f1_score:.2f}")
