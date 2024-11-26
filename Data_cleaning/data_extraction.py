import pandas as pd
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

# Current working directory
current_dir = os.getcwd()

# Define relative path
relative_path = "raw_data/release_in_the_wild/"

# Define absolute path
absolute_path = os.path.join(current_dir, relative_path)

# Create an empty DataFrame
columns = ['file_index', 'audio_length']
df_audio_length = pd.DataFrame(columns=columns)

for file_num in range(0, 31778):
    # Load audio file
    audio_file = os.path.join(absolute_path, f'{file_num}.wav')
    y, sr = librosa.load(audio_file, sr=16000)

    # Calculate duration of each audio file in seconds
    duration = round(len(y) / sr, 2)

    # Create a new row
    new_row = pd.DataFrame({'file_index': [file_num], 'audio_length': [duration]})

    # Concatenate the new row to the existing DataFrame
    df_audio_length = pd.concat([df_audio_length, new_row], ignore_index=True)

### Join with CSV

# Import CSV
csv_path=os.path.join(absolute_path, 'meta.csv')
csv_df=pd.read_csv(csv_path)

# Add file index
csv_df['file_index']=csv_df['file'].str.strip('.wav').astype(int)

# Merge with audio_file DF
df_merged=pd.merge(csv_df, df_audio_length, on='file_index', how='inner')
print(df_merged)
