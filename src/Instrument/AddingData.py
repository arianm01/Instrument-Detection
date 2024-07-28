import os
import random
from tqdm import tqdm

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment


def create_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def slice_audio(audio_path, output_dir, segment_length_ms):
    """
    Slice audio files into segments and save them.

    :param audio_path: Directory containing the original audio files.
    :param output_dir: Directory where the audio segments will be saved.
    :param segment_length_ms: Length of each audio segment in milliseconds.
    """
    classes = os.listdir(audio_path)

    # Create output directories for each class
    for instrument in classes:
        instrument_output_dir = os.path.join(output_dir, instrument)
        create_directory(instrument_output_dir)

    # Process each audio file
    for file in os.listdir(audio_path):
        audio = AudioSegment.from_file(os.path.join(audio_path, file))
        num_segments = len(audio) // segment_length_ms

        # Split and save each segment
        for i in range(num_segments):
            start_ms = i * segment_length_ms
            end_ms = start_ms + segment_length_ms
            segment = audio[start_ms:end_ms]
            segment_filename = os.path.join(output_dir, file, f"{file}_{i + 1}.mp3")
            segment.export(segment_filename, format="mp3")


def clean_audio(audio_files, audio_path, output_dir, threshold=0.001, sample_rate=16000):
    """
    Clean audio files by removing silent parts.

    :param audio_files: List of audio file names.
    :param audio_path: Directory containing the original audio files.
    :param output_dir: Directory where the cleaned audio files will be saved.
    :param threshold: Amplitude threshold for silence detection.
    :param sample_rate: Sampling rate of the audio files.
    """
    for file in tqdm(audio_files):
        file_path = os.path.join(audio_path, file)
        signal, sr = librosa.load(file_path, sr=sample_rate)

        y = pd.Series(signal).apply(np.abs)
        y_mean = y.rolling(window=int(sr / 10), min_periods=1, center=True).mean()
        mask = y_mean > threshold

        cleaned_signal = signal[mask]
        output_file_path = os.path.join(output_dir, file)
        sf.write(output_file_path, cleaned_signal, sr)


def save_array_to_file(array, filename):
    """
    Save an array to a text file.

    :param array: Array to save.
    :param filename: Path to the output text file.
    """
    create_directory(os.path.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as file:
        for item in array:
            file.write(f"{item}\n")


def split_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, validation, and test sets.

    :param input_dir: Directory containing Instrument folders with files.
    :param output_dir: Directory where the splits will be saved.
    :param train_ratio: Ratio of training data.
    :param val_ratio: Ratio of validation data.
    :param test_ratio: Ratio of test data.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"

    create_directory(output_dir)
    splits = ['train', 'val', 'test']
    for split in splits:
        create_directory(os.path.join(output_dir, split))

    for instrument in os.listdir(input_dir):
        instrument_dir = os.path.join(input_dir, instrument)
        if not os.path.isdir(instrument_dir):
            continue

        files = os.listdir(instrument_dir)
        random.shuffle(files)

        total_files = len(files)
        train_end = int(train_ratio * total_files)
        val_end = train_end + int(val_ratio * total_files)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        save_array_to_file(train_files, os.path.join(output_dir, 'train', f'{instrument}.txt'))
        save_array_to_file(val_files, os.path.join(output_dir, 'val', f'{instrument}.txt'))
        save_array_to_file(test_files, os.path.join(output_dir, 'test', f'{instrument}.txt'))


def main():
    audio_path = ("../../../../archive/Persian Classical Music Instrument Recognition (PCMIR) Database/Persian "
                  "Classical Music Instrument Recognition (PCMIR) Database/Ud")
    output_dir = "audio_segments_test"
    segment_length_ms = 1000  # Segment length in milliseconds
    sample_rate = 16000

    create_directory(output_dir)

    slice_audio(audio_path, output_dir, segment_length_ms)

    # Clean audio files
    audio_files = os.listdir(audio_path)
    clean_audio(audio_files, audio_path, output_dir, threshold=0.001, sample_rate=sample_rate)

    # Split data
    input_directory = './audio_segments_test'
    split_output_directory = './splits'
    split_data(input_directory, split_output_directory)


if __name__ == "__main__":
    main()
