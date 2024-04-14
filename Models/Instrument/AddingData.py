from pydub import AudioSegment
import os

# Load the audio file
audio_path = "../../../../archive/Instruments/test"
files = os.listdir(audio_path)
segment_length_ms = 5000  # 5 seconds
# Create a directory for the audio segments if it doesn't already exist
output_dir = "audio_segments_test"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in files:
    audio = AudioSegment.from_file(audio_path+'/'+file)
    # Define the length of each segment in milliseconds

    # Calculate the number of segments
    num_segments = len(audio) // segment_length_ms

    # Split the audio and save each segment
    for i in range(num_segments):
        start_ms = i * segment_length_ms
        end_ms = start_ms + segment_length_ms
        segment = audio[start_ms:end_ms]
        segment_filename = f"{file}_{i + 1}.mp3"
        segment_path = os.path.join(output_dir, segment_filename)
        segment.export(segment_path, format="mp3")

    print(f"Audio file split into {num_segments} segments and saved to '{output_dir}' directory.")
