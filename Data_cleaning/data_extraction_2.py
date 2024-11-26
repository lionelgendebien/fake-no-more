import pandas as pd
import librosa
import numpy as np
import os

# Current working directory
current_dir = os.getcwd()

# Define relative path
relative_path = "raw_data/release_in_the_wild/"

# Define absolute path
absolute_path = os.path.join(current_dir, relative_path)

# Create a DataFrame for each feature
mfccs_df = pd.DataFrame()
chroma_stft_df = pd.DataFrame()
spectral_contrast_df = pd.DataFrame()
zero_crossing_rate_df = pd.DataFrame()
spectral_bandwidth_df = pd.DataFrame()
spectral_rolloff_df = pd.DataFrame()

for file_num in range(0, 2):
    # Load audio file
    audio_file = os.path.join(absolute_path, f'{file_num}.wav')
    y, sr = librosa.load(audio_file, sr=16000)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Create DataFrames for each feature
    mfccs_temp_df = pd.DataFrame(mfccs.T, columns=[f'mfcc_{i+1}' for i in range(mfccs.shape[0])])
    mfccs_temp_df['file_index'] = file_num

    chroma_temp_df = pd.DataFrame(chroma_stft.T, columns=['chroma'])
    chroma_temp_df['file_index'] = file_num

    spectral_contrast_temp_df = pd.DataFrame(spectral_contrast.T, columns=[f'spectral_contrast_band_{i+1}' for i in range(spectral_contrast.shape[0])])
    spectral_contrast_temp_df['file_index'] = file_num

    zero_crossing_temp_df = pd.DataFrame(zero_crossing_rate.T, columns=['zero_crossing_rate'])
    zero_crossing_temp_df['file_index'] = file_num

    spectral_bandwidth_temp_df = pd.DataFrame(spectral_bandwidth.T, columns=['spectral_bandwidth'])
    spectral_bandwidth_temp_df['file_index'] = file_num

    spectral_rolloff_temp_df = pd.DataFrame(spectral_rolloff.T, columns=['spectral_rolloff'])
    spectral_rolloff_temp_df['file_index'] = file_num

    # Concatenate to main DataFrames
    mfccs_df = pd.concat([mfccs_df, mfccs_temp_df], ignore_index=True)
    chroma_stft_df = pd.concat([chroma_stft_df, chroma_temp_df], ignore_index=True)
    spectral_contrast_df = pd.concat([spectral_contrast_df, spectral_contrast_temp_df], ignore_index=True)
    zero_crossing_rate_df = pd.concat([zero_crossing_rate_df, zero_crossing_temp_df], ignore_index=True)
    spectral_bandwidth_df = pd.concat([spectral_bandwidth_df, spectral_bandwidth_temp_df], ignore_index=True)
    spectral_rolloff_df = pd.concat([spectral_rolloff_df, spectral_rolloff_temp_df], ignore_index=True)

# Merge all feature DataFrames on 'file_index'
merged_df = (mfccs_df
             .merge(chroma_stft_df, on='file_index', how='outer')
             .merge(spectral_contrast_df, on='file_index', how='outer')
             .merge(zero_crossing_rate_df, on='file_index', how='outer')
             .merge(spectral_bandwidth_df, on='file_index', how='outer')
             .merge(spectral_rolloff_df, on='file_index', how='outer'))

# Print final DataFrame
print(merged_df)
