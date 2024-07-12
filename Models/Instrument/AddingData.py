import random
import shutil

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import os
import soundfile as sf

from tqdm import tqdm


def slice_audio():
    # classes = os.listdir(audio_path)
    # print(classes)
    # #
    # for instrument in classes:
    #     print(instrument)
    #     if not os.path.exists(output_dir + '/' + instrument):
    #         os.makedirs(output_dir + '/' + instrument)
    files = os.listdir(os.path.join(audio_path))
    print(files)
    for file in files:
        audio = AudioSegment.from_file(os.path.join(audio_path, file))
        # Define the length of each segment in milliseconds
        # Calculate the number of segments
        num_segments = len(audio) // segment_length_ms

        # Split the audio and save each segment
        for i in range(num_segments):
            start_ms = i * segment_length_ms
            end_ms = start_ms + segment_length_ms
            segment = audio[start_ms:end_ms]
            segment_filename = f"{output_dir}/Oud/{file}_{i + 1}.mp3"
            segment.export(segment_filename, format="mp3")

        print(f"Audio file split into {num_segments} segments and saved to '{output_dir}' directory.")


def clean_audio(audio_files):
    for file in tqdm(audio_files):
        mask = []
        file_path = os.path.join(audio_path + '/' + file)
        signal, sample_rate = librosa.load(file_path)
        y = pd.Series(signal).apply(np.abs)
        y_mean = y.rolling(window=int(sample_rate / 10),
                           min_periods=1,
                           center=True).mean()
        for mean in y_mean:
            if mean > 0.001:
                mask.append(True)
            else:
                mask.append(False)
        print(len(mask))
        cleaned_signal = signal[mask]

        # Saving the cleaned signal
        output_file_path = os.path.join('audio_segments_test', file)
        sf.write(output_file_path, cleaned_signal, 16000)
        print(cleaned_signal.shape)


# Load the audio file
# audio_path = "../../../../archive/Persian Classical Music Instrument Recognition (PCMIR) Database/Persian Classical Music Instrument Recognition (PCMIR) Database/Ud"
# files = os.listdir(audio_path)
# segment_length_ms = 1000  # 5 seconds
# # Create a directory for the audio segments if it doesn't already exist
# output_dir = "audio_segments_test/output"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# slice_audio()


def save_array_to_file(array, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in array:
            file.write(f"{item}\n")


def split_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, validation, and test sets.

    :param input_dir: Directory containing instrument folders with files.
    :param output_dir: Directory where the splits will be saved.
    :param train_ratio: Ratio of training data.
    :param val_ratio: Ratio of validation data.
    :param test_ratio: Ratio of test data.
    """
    # Ensure the ratios sum to 1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0, rtol=1e-09, atol=1e-09), "Ratios must sum to 1"

    # Create output directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each instrument directory
    for instrument in os.listdir(input_dir):
        instrument_dir = os.path.join(input_dir, instrument)
        if not os.path.isdir(instrument_dir):
            continue

        # List all files in the current instrument directory
        files = os.listdir(instrument_dir)
        random.shuffle(files)  # Shuffle files for randomness

        # Compute split indices
        total_files = len(files)
        train_end = int(train_ratio * total_files)
        val_end = train_end + int(val_ratio * total_files)

        # Split files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        save_array_to_file(train_files, f'splits/train/{instrument}.txt')
        save_array_to_file(val_files, f'splits/val/{instrument}.txt')
        save_array_to_file(test_files, f'splits/test/{instrument}.txt')

        print(f"Instrument '{instrument}' - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# # Example usage
input_directory = './audio_segments_test'
output_directory = '/splits'
split_data(input_directory, output_directory)
