import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import os
import soundfile as sf

from tqdm import tqdm


def slice_audio(instrument):
    # classes = os.listdir(audio_path)
    # print(classes)

    # for instrument in classes:
    print(instrument)
    if not os.path.exists(output_dir + '/' + instrument):
        os.makedirs(output_dir + '/' + instrument)
    files = os.listdir(os.path.join(audio_path, instrument))
    for file in files:
        audio = AudioSegment.from_file(os.path.join(audio_path, instrument) + '/' + file)
        # Define the length of each segment in milliseconds
        # Calculate the number of segments
        num_segments = len(audio) // segment_length_ms

        # Split the audio and save each segment
        for i in range(num_segments):
            start_ms = i * segment_length_ms
            end_ms = start_ms + segment_length_ms
            segment = audio[start_ms:end_ms]
            segment_filename = f"{output_dir}/{instrument}/{file}_{i + 1}.mp3"
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
audio_path = "../../../../archive/Instruments"
files = os.listdir(audio_path)
segment_length_ms = 1000  # 5 seconds
# Create a directory for the audio segments if it doesn't already exist
output_dir = "audio_segments_test"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

slice_audio('Ney Anban')
# clean_audio(files)
