import os
import random
from tqdm import tqdm

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

# --- CONFIGURATION ---
DATASET_DIR = 'audio_dataset'
OUTPUT_DIR = 'split_chunks'
CHUNK_MS = 1000  # 1 second
RANDOM_SEED = 42
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

random.seed(RANDOM_SEED)

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def get_duration_ms(file_path):
    return len(AudioSegment.from_mp3(file_path))

def slice_and_save(audio_path, out_dir, base_name):
    audio = AudioSegment.from_wav(audio_path)
    for i in range(0, len(audio), CHUNK_MS):
        chunk = audio[i:i + CHUNK_MS]
        if len(chunk) < CHUNK_MS:
            continue
        chunk_name = f"{base_name}_chunk_{i // CHUNK_MS:04}.wav"
        chunk.export(os.path.join(out_dir, chunk_name), format="wav")

def slice_audio(audio_path, output_dir, segment_length_ms):
    classes = os.listdir(audio_path)
    for cls in classes:
        class_input_path = os.path.join(audio_path, cls)
        class_output_path = os.path.join(output_dir, cls)
        create_directory(class_output_path)

        for file in os.listdir(class_input_path):
            file_path = os.path.join(class_input_path, file)
            audio = AudioSegment.from_file(file_path)
            num_segments = len(audio) // segment_length_ms

            for i in range(num_segments):
                start_ms = i * segment_length_ms
                end_ms = start_ms + segment_length_ms
                segment = audio[start_ms:end_ms]
                segment_filename = os.path.join(class_output_path, f"{os.path.splitext(file)[0]}_{i + 1}.mp3")
                segment.export(segment_filename, format="mp3")

def clean_audio(audio_files, audio_path, output_dir, threshold=0.001, sample_rate=16000):
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
    create_directory(os.path.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(f"{item}\n" for item in array)

def split_data(input_dir, output_dir, ratios):
    assert np.isclose(sum(ratios.values()), 1.0), "Ratios must sum to 1"

    for split in ratios:
        create_directory(os.path.join(output_dir, split))

    for cls in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        files = os.listdir(class_dir)
        random.shuffle(files)

        total = len(files)
        train_end = int(ratios['train'] * total)
        val_end = train_end + int(ratios['val'] * total)

        split_files = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        for split, file_list in split_files.items():
            save_array_to_file(file_list, os.path.join(output_dir, split, f'{cls}.txt'))

def duration_aware_segmentation(audio_path):
    for cls in os.listdir(audio_path):
        class_path = os.path.join(audio_path, cls)
        files_with_durations = []

        for f in os.listdir(class_path):
            full_path = os.path.join(class_path, f)
            duration = get_duration_ms(full_path)
            files_with_durations.append((f, duration))

        total_duration = sum(d for _, d in files_with_durations)
        target_durations = {k: total_duration * v for k, v in SPLIT_RATIOS.items()}
        current_durations = {k: 0 for k in SPLIT_RATIOS}
        splits = {k: [] for k in SPLIT_RATIOS}

        for f, dur in sorted(files_with_durations, key=lambda x: -x[1]):
            best_split = min(current_durations, key=lambda s: current_durations[s] / target_durations[s])
            splits[best_split].append(f)
            current_durations[best_split] += dur

        for split, file_list in splits.items():
            split_dir = os.path.join(OUTPUT_DIR, split)
            create_directory(split_dir)
            save_array_to_file(file_list, os.path.join(split_dir, f'{cls}.txt'))

            for f in file_list:
                full_path = os.path.join(DATASET_DIR, cls, f)
                base = os.path.splitext(f)[0]
                slice_and_save(full_path, split_dir, base)

def main():
    audio_path = "../../../../archive/Instruments"
    output_dir = "audio_segments_test"
    sample_rate = 16000

    create_directory(output_dir)
    slice_audio(audio_path, output_dir, CHUNK_MS)

    audio_files = os.listdir(audio_path)
    # clean_audio(audio_files, audio_path, output_dir, threshold=0.001, sample_rate=sample_rate)

    split_data(output_dir, './splits', SPLIT_RATIOS)
    # Optionally call: duration_aware_segmentation(output_dir)

if __name__ == "__main__":
    main()
