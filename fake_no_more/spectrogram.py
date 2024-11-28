import pandas as pd
import librosa
import os
import numpy as np
import sys

# Current working directory
current_dir = os.getcwd()

# Define relative path
relative_path = "raw_data/release_in_the_wild/"

# Define absolute path
absolute_path = os.path.join(current_dir, relative_path)

# Create load and preprocess function
SAMPLE_RATE = 16000
DURATION = 5
MAX_LEN = SAMPLE_RATE * DURATION

# Apply filters on audio files
csv_path="raw_data/data_mapping.csv"
mapping_file=pd.read_csv(csv_path)

    # Filter 1: keep audio files of min 5''
filter_1=mapping_file[mapping_file['audio_length'] >= 5]

    # Filter 2:keep audio files whose total cumul length per speaker per label
    # (i.e. Fake or Real) does not exceed 40'' (NB: only 7 out of 54 speakers
    # have total audio length of either fake or real below 40'')
filter_2=filter_1.sort_values(by=['speaker', 'label', 'audio_length'], ascending=[True, True, True])
filter_2['cum_audio_length'] = filter_2.groupby(['speaker', 'label'])['audio_length'].cumsum()
filter_2 = filter_2[filter_2['cum_audio_length'] <= 40]
final_filtering=filter_2[['file_index', 'label']]

mel_spectrograms = []

for file_num in final_filtering.file_index:
    # Load audio file
    audio_file = os.path.join(absolute_path, f'{file_num}.wav')
    y, _ = librosa.load(audio_file, sr=SAMPLE_RATE)

    if len(y) >= MAX_LEN:
        # Find the starting index for the 1-second extract at the middle
        middle_start = int((len(y) - SAMPLE_RATE * DURATION) / 2)
        # Extract the 1-second segment from the middle
        y = y[middle_start:middle_start + SAMPLE_RATE * DURATION]

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE)

        mel_spectrograms.append(mel_spectrogram)

X = np.stack(mel_spectrograms)
