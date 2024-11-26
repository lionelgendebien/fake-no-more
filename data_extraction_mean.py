import pandas as pd
import librosa
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq

# Current working directory
current_dir = os.getcwd()

# Define relative path
relative_path = "raw_data/raw_data/release_in_the_wild/"

# Define absolute path
absolute_path = os.path.join(current_dir, relative_path)

# Create a DataFrame for each feature
mfccs_df = pd.DataFrame()
chroma_stft_df = pd.DataFrame()
spectral_contrast_df = pd.DataFrame()
zero_crossing_rate_df = pd.DataFrame()
spectral_bandwidth_df = pd.DataFrame()
spectral_rolloff_df = pd.DataFrame()

SAMPLE_RATE = 16000
DURATION = 1
MAX_LEN = SAMPLE_RATE * DURATION


for file_num in range(0, 3001):
    # Load audio file
    audio_file = os.path.join(absolute_path, f'{file_num}.wav')
    y, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)

    if len(y) >= MAX_LEN:
        # Find the starting index for the 1-second extract at the middle
        middle_start = int((len(y) - SAMPLE_RATE * DURATION) / 2)
        # Extract the 1-second segment from the middle
        y = y[middle_start:middle_start + SAMPLE_RATE * DURATION]

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_bands=6)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)

    # Take the mean of each feature across time (axis=1)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma_stft, axis=1)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate, axis=1)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth, axis=1)
    spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)

    # Create DataFrames for each feature (now with 1 row per file)
    mfccs_temp_df = pd.DataFrame([mfccs_mean], columns=[f'mfcc_{i+1}' for i in range(mfccs_mean.shape[0])])
    mfccs_temp_df['index'] = file_num

    chroma_temp_df = pd.DataFrame([chroma_mean], columns=['chroma'])
    chroma_temp_df['index'] = file_num

    spectral_contrast_temp_df = pd.DataFrame([spectral_contrast_mean], columns=[f'spectral_contrast_band_{i+1}' for i in range(spectral_contrast_mean.shape[0])])
    spectral_contrast_temp_df['index'] = file_num

    zero_crossing_temp_df = pd.DataFrame([zero_crossing_rate_mean], columns=['zero_crossing_rate'])
    zero_crossing_temp_df['index'] = file_num

    spectral_bandwidth_temp_df = pd.DataFrame([spectral_bandwidth_mean], columns=['spectral_bandwidth'])
    spectral_bandwidth_temp_df['index'] = file_num

    spectral_rolloff_temp_df = pd.DataFrame([spectral_rolloff_mean], columns=['spectral_rolloff'])
    spectral_rolloff_temp_df['index'] = file_num

    # Concatenate to main DataFrames
    mfccs_df = pd.concat([mfccs_df, mfccs_temp_df], ignore_index=True)
    chroma_stft_df = pd.concat([chroma_stft_df, chroma_temp_df], ignore_index=True)
    spectral_contrast_df = pd.concat([spectral_contrast_df, spectral_contrast_temp_df], ignore_index=True)
    zero_crossing_rate_df = pd.concat([zero_crossing_rate_df, zero_crossing_temp_df], ignore_index=True)
    spectral_bandwidth_df = pd.concat([spectral_bandwidth_df, spectral_bandwidth_temp_df], ignore_index=True)
    spectral_rolloff_df = pd.concat([spectral_rolloff_df, spectral_rolloff_temp_df], ignore_index=True)

# Merge all feature DataFrames on 'index'
merged_df = (mfccs_df
             .merge(chroma_stft_df, on='index', how='outer')
             .merge(spectral_contrast_df, on='index', how='outer')
             .merge(zero_crossing_rate_df, on='index', how='outer')
             .merge(spectral_bandwidth_df, on='index', how='outer')
             .merge(spectral_rolloff_df, on='index', how='outer'))

# merged_df = merged_df.merge(data[['file_index', 'label']], left_on='index', right_on='file_index', how='left')

# merged_df = merged_df.drop(columns=['file_index'])

# print(merged_df)
# merged_df.to_parquet('data_mean.parquet', engine='pyarrow')

# # Import the label (Fake or Real)
csv_path=os.path.join(absolute_path, "meta.csv")
mapping_file=pd.read_csv(csv_path)
mapping_file['index']=mapping_file.file.str.replace('.wav', '', regex=False).astype(int)
mapping_file=mapping_file.drop(columns='file')
print(mapping_file)

# Merge to retrieve label
data_df = merged_df.merge(mapping_file, on='index',how='inner')

# Print final DataFrame
print(data_df)

# Export as Parquet file
data_df.to_parquet('data_mean.parquet', engine='pyarrow')
