import os
from collections import Counter

import librosa
import numpy as np
from pydub import AudioSegment
from tensorflow.keras.models import load_model


def preprocess_audio(audio_path, segment_duration=4, n_mfcc=13):
    """
    Load an audio file, slice it into 5-second segments, and extract MFCC features for each segment.
    """
    # Load the full audio file
    signal, sample_rate = librosa.load(audio_path)

    # Calculate the number of samples per segment
    samples_per_segment = int(sample_rate * segment_duration)

    # Determine the total number of segments
    num_segments = int(np.ceil(len(signal) / samples_per_segment))

    segments = []

    for segment in range(num_segments):
        # Calculate the start and end sample for the current segment
        start_sample = samples_per_segment * segment
        end_sample = start_sample + samples_per_segment
        print("Start sample", start_sample, end_sample)
        # If we're not at the end of the signal
        if end_sample <= len(signal):
            # Extract the segment
            segment_signal = signal[start_sample:end_sample]

            # Compute the MFCCs
            mfcc = librosa.feature.mfcc(y=segment_signal, sr=sample_rate, n_fft=2048, hop_length=512, n_mfcc=n_mfcc)
            segments.append(mfcc.T)

    return np.array(segments)


def predict(audio_path, model_path='../../model_best_CNN_1.h5'):
    """
    Predict the class of each segment of an audio file.
    """
    # Preprocess the audio to get its segments
    segments = preprocess_audio(audio_path)

    # Add a new axis to match the input shape of the model
    segments = segments[..., np.newaxis]

    # Load the trained model
    model = load_model(model_path)

    # Predict
    predictions = model.predict(segments)

    # Convert predictions to class labels if necessary
    return np.argmax(predictions, axis=1)


# Example usage
audio_path = '../../Test/2_0_01_00.mp3'
predicted_labels = predict(audio_path)
classes = os.listdir('../../input')
print(predicted_labels)
for labels in predicted_labels:
    print(classes[labels])

label_counts = Counter(predicted_labels)

# Print the count of each class
for label, count in label_counts.items():
    class_name = classes[label]
    print(f"Class '{class_name}' appears {count} times in predictions.")