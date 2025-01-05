import os
import librosa
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model

from src.Instrument.Contrastive import SupervisedContrastiveLoss
from src.Instrument.ContrastiveLearning import generate_embeddings
from src.main.TransformerModel import model
from src.main.main import get_model_feature
from src.utility.InstrumentDataset import create_sliding_windows, compute_mel_spectrogram

# Constants
LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
LABEL_MAPPING_dastgah = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
All_LABEL_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14}
Dastgahs = ['Shour', 'Segah', 'Mahour', 'Homayoun', 'RastPanjgah', 'Nava', 'Chahargah']
CLASSES = ['Tar', 'Kamancheh', 'Santur', 'Setar', 'Ney', '']
All_CLASSES = ['Daf', 'Divan', 'Dutar', 'Gheychak', 'Kamancheh', 'Ney', 'Ney Anban', 'Oud', 'Qanun', 'Rubab', 'Santur',
               'Setar', 'Tanbour', 'Tar', 'Tonbak', '']
mapping = {i: All_CLASSES.index(cls) for i, cls in enumerate(CLASSES)}
IsDastgahDetection = True


def class_mapping(predictions):
    labels = []
    for cl in predictions:
        try:
            labels.append(CLASSES.index(All_CLASSES[cl]))
        except ValueError:
            labels.append(5)
    return np.array(labels)


def load_files(audio_path):
    """Loads the list of files to process."""
    try:
        with open(audio_path, 'r') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print("File not found.")
        return []


def predict(audio_path, model, models, model_base, contrastive=False):
    """Predicts the labels for the audio file located at audio_path."""
    segments = preprocess_audio(audio_path)
    if segments.size == 0:
        return np.array([])
    segments = segments[..., np.newaxis]
    # return class_mapping(predict_segments(segments, model, models, model_base, contrastive))
    return predict_segments(segments, model, models, model_base, contrastive)


def main():
    audio_path = '../../../../archive/NavaDataset'
    path = '../../output/Dastgah/2 sec/ensemble.keras'
    files = load_files(audio_path + '/test.txt')
    models_addr = ['../../output/Dastgah/1 sec/model_best_classifier_1.keras',
                   '../../output/Dastgah/1 sec/model_best_classifier_2.keras',
                   '../../output/Dastgah/1 sec/model_best_classifier_3.keras',
                   '../../output/Dastgah/1 sec/model_best_classifier_4.keras',
                   '../../output/Dastgah/1 sec/model_best_classifier_5.keras']
    model, models, model_base = load_models(path, models_addr, models_addr[0])
    contrastive = False

    true_labels, predicted_labels = process_files(files, audio_path, model, models, model_base, contrastive)
    evaluate_predictions(true_labels, predicted_labels)


def preprocess_audio(audio_path, segment_duration=2, step_size=None):
    """Preprocesses the audio file into mel_spectrogram segments."""
    try:
        if step_size is not None:
            signal, sample_rate = librosa.load(audio_path)
            windows = create_sliding_windows(signal, segment_duration, step_size)
            mel_spectrograms = []

            for i, window in enumerate(windows):
                mel_spectrogram = compute_mel_spectrogram(window, sample_rate)
                mel_spectrograms.append(mel_spectrogram)
            return mel_spectrograms

        signal, sample_rate = librosa.load(audio_path)
        samples_per_segment = int(sample_rate * segment_duration)
        num_segments = int(np.ceil(len(signal) / samples_per_segment))
        segments = []

        for segment in range(num_segments):
            start_sample = samples_per_segment * segment
            end_sample = start_sample + samples_per_segment
            if end_sample <= len(signal):
                segment_signal = signal[start_sample:end_sample]
                mel_spectrogram = compute_mel_spectrogram(segment_signal, sample_rate)
                segments.append(mel_spectrogram)
        return np.array(segments)
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return np.array([])


def predict_segments(segments, model, models, model_base, contrastive=False):
    """Predicts the labels for the given segments using the appropriate model."""
    try:
        if contrastive:
            predictions = predict_contrastive(segments, model)
        else:
            meta = get_model_feature(segments, models)
            predictions = model.predict(meta)
            # predictions = model_base.predict(segments)
        return np.argmax(predictions, axis=1)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return np.array([])


def predict_contrastive(segments, model_base):
    """Generates predictions using the contrastive learning model."""
    embeddings = generate_embeddings(model_base, segments, 'model')
    return model.predict(embeddings)


def extract_label(file_name):
    """Extracts the label from the file name."""
    try:
        return LABEL_MAPPING_dastgah[int(file_name[2])]
    except (IndexError, ValueError, KeyError):
        return None


def load_models(addr_model, addr_models, addr_model_base):
    """Loads the required models."""
    model = load_model(addr_model)
    models = [load_model(addr_models[0]), load_model(addr_models[1]), load_model(addr_models[2]),
              load_model(addr_models[3]), load_model(addr_models[4])]
    model_base = load_model(addr_model_base, custom_objects={
        'SupervisedContrastiveLoss': SupervisedContrastiveLoss
    })
    # models = []
    # model_base = 2
    return model, models, model_base


def process_files(files, audio_path, model, models, model_base, contrastive=False):
    """Processes the list of files and returns true and predicted labels."""
    true_labels = []
    predicted_labels = []

    for file in files:
        true_label = extract_label(file)
        if true_label is not None:
            predicted_label = predict(os.path.join(audio_path, 'Data', file + '.mp3'), model, models, model_base,
                                      contrastive)
            true_labels.extend([true_label] * len(predicted_label))
            predicted_labels.extend(predicted_label)
            display_predictions(file, predicted_label)

    return true_labels, predicted_labels


def display_predictions(file, predicted_label):
    """Displays the prediction results.txt."""
    if predicted_label.size > 0:
        label_counts = Counter(predicted_label)
        for label, count in label_counts.items():
            if IsDastgahDetection:
                class_name = Dastgahs[label]
            else:
                class_name = CLASSES[label]
            print(f"Class '{class_name}' appears {count} times in predictions for {file}.")
    else:
        print(f"No predictions for {file}.")


def evaluate_predictions(true_labels, predicted_labels, extended_labels=False):
    """Evaluates and prints the prediction performance metrics."""
    true_labels = [int(x) for x in true_labels]
    print("True Labels:", true_labels)
    print("Predicted Labels:", predicted_labels)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print('Confusion matrix \n', conf_matrix)

    class_accuracies = calculate_class_accuracies(conf_matrix, extended_labels)
    display_class_accuracies(class_accuracies, extended_labels)

    overall_accuracy = calculate_overall_accuracy(conf_matrix)
    print(f"Total Accuracy: {overall_accuracy * 100:.2f}%")

    f1_scores = f1_score(true_labels, predicted_labels, average=None)
    macro_f1_score = f1_score(true_labels, predicted_labels, average='weighted')

    precision_scores = precision_score(true_labels, predicted_labels, average=None)
    macro_precision_score = precision_score(true_labels, predicted_labels, average='weighted')

    recall_scores = recall_score(true_labels, predicted_labels, average=None)
    macro_recall_score = recall_score(true_labels, predicted_labels, average='weighted')

    display_f1_scores(f1_scores, macro_f1_score, extended_labels)
    display_precision_recall(precision_scores, recall_scores, macro_precision_score, macro_recall_score,
                             extended_labels)


def calculate_class_accuracies(conf_matrix, extended):
    """Calculates accuracies for each class."""
    if extended:
        labels = All_LABEL_MAPPING
    else:
        labels = LABEL_MAPPING_dastgah

    class_accuracies = {}
    for i, class_label in enumerate(labels.values()):
        true_positives = conf_matrix[i, i]
        total_actual = conf_matrix[i, :].sum()  # Row sum instead of column sum
        accuracy = true_positives / total_actual if total_actual > 0 else 0
        class_accuracies[i] = accuracy
    return class_accuracies


def display_class_accuracies(class_accuracies, extended):
    """Displays accuracies for each class."""
    if extended:
        classes = All_CLASSES
    elif IsDastgahDetection:
        classes = Dastgahs
    else:
        classes = CLASSES

    for label, acc in class_accuracies.items():
        print(f"Accuracy for class {classes[label]}: {acc * 100:.2f}%")


def calculate_overall_accuracy(conf_matrix):
    """Calculates overall accuracy."""
    total_true_positives = np.trace(conf_matrix)
    total_predictions = conf_matrix.sum()
    return total_true_positives / total_predictions


def display_f1_scores(f1_scores, macro_f1_score, extended):
    """Displays F1 scores for each class and the macro-average F1 score."""
    if extended:
        classes = All_CLASSES
        labels = All_LABEL_MAPPING
    elif IsDastgahDetection:
        classes = Dastgahs
        labels = LABEL_MAPPING_dastgah
    else:
        classes = CLASSES
        labels = LABEL_MAPPING

    for i, label in enumerate(labels.values()):
        print(f"F1 Score for class {classes[label]}: {f1_scores[i]:.2f}")
    print(f"Macro-average F1 Score: {macro_f1_score:.2f}")


def display_precision_recall(precision_scores, recall_scores, macro_precision_score, macro_recall_score, extended):
    if extended:
        classes = All_CLASSES
        labels = All_LABEL_MAPPING
    elif IsDastgahDetection:
        classes = Dastgahs
        labels = LABEL_MAPPING_dastgah
    else:
        classes = CLASSES
        labels = LABEL_MAPPING
    """Displays precision and recall scores for each class and the macro-average scores."""
    for i, label in enumerate(labels.values()):
        print(f"Precision for class {classes[label]}: {precision_scores[i]:.2f}")
        print(f"Recall for class {classes[label]}: {recall_scores[i]:.2f}")
    print(f"Macro-average Precision: {macro_precision_score:.2f}")
    print(f"Macro-average Recall: {macro_recall_score:.2f}")


if __name__ == "__main__":
    main()
