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

# Create a DataFrame for each feature
mfccs_df = pd.DataFrame()
chroma_stft_df = pd.DataFrame()
spectral_contrast_df = pd.DataFrame()
zero_crossing_rate_df = pd.DataFrame()
spectral_bandwidth_df = pd.DataFrame()
spectral_rolloff_df = pd.DataFrame()

# Create load and preprocess function
SAMPLE_RATE = 16000
DURATION = 5
MAX_LEN = SAMPLE_RATE * DURATION

# Apply filters on audio files
csv_path="raw_data/data_mapping.csv"
mapping_file=pd.read_csv(csv_path)

# Filter 1: Augment each individual audio of < 5'' to 5''
mapping_file_augmented=mapping_file
mapping_file_augmented['audio_length_adjusted'] = mapping_file_augmented['audio_length'].apply(lambda x: 5 if x < 5 else x)

# Filter 2: limit audio files to cum sum of 200'' after restricting each audio to 5''
mapping_file_augmented['audio_length_adjusted'] = mapping_file_augmented['audio_length_adjusted'].apply(lambda x: 5 if x > 5 else x)
mapping_file_augmented = mapping_file_augmented.sort_values(by=['speaker', 'label', 'audio_length'], ascending=[True, True, False])
mapping_file_augmented['cum_audio_length_adjusted'] = mapping_file_augmented.groupby(['speaker', 'label'])['audio_length_adjusted'].cumsum()
mapping_file_augmented_filtered = mapping_file_augmented[mapping_file_augmented['cum_audio_length_adjusted'] <= 200]
final_filtering=mapping_file_augmented_filtered[['file_index', 'label']]

for file_num in final_filtering.file_index:
    # Load audio file
    audio_file = os.path.join(absolute_path, f'{file_num}.wav')
    y, _ = librosa.load(audio_file, sr=SAMPLE_RATE)

    if len(y) < MAX_LEN:  # If the audio is shorter than 5 seconds
        # Repeat the audio until it exceeds or equals 5 seconds
        repeats = MAX_LEN // len(y) + 1  # Calculate the number of repeats needed
        y = np.tile(y, repeats)[:MAX_LEN]  # Repeat and trim to exactly 5 seconds
    else:
        # Extract a 5-second segment from the mid-point
        start_idx = (len(y) - MAX_LEN) // 2  # Calculate starting index for mid-point extraction
        y = y[start_idx:start_idx + MAX_LEN]  # Extract the middle 5 seconds

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_bands=6)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)

    # Create DataFrames for each feature
    mfccs_temp_df = pd.DataFrame(mfccs.T, columns=[f'mfcc_{i+1}' for i in range(mfccs.shape[0])])
    mfccs_temp_df['file_index'] = file_num

    chroma_temp_df = pd.DataFrame(chroma_stft.T, columns=['chroma'])

    spectral_contrast_temp_df = pd.DataFrame(spectral_contrast.T, columns=[f'spectral_contrast_band_{i+1}' for i in range(spectral_contrast.shape[0])])

    zero_crossing_temp_df = pd.DataFrame(zero_crossing_rate.T, columns=['zero_crossing_rate'])

    spectral_bandwidth_temp_df = pd.DataFrame(spectral_bandwidth.T, columns=['spectral_bandwidth'])

    spectral_rolloff_temp_df = pd.DataFrame(spectral_rolloff.T, columns=['spectral_rolloff'])

    # Concatenate to main DataFrames
    mfccs_df = pd.concat([mfccs_df, mfccs_temp_df], ignore_index=True)
    chroma_stft_df = pd.concat([chroma_stft_df, chroma_temp_df], ignore_index=True)
    spectral_contrast_df = pd.concat([spectral_contrast_df, spectral_contrast_temp_df], ignore_index=True)
    zero_crossing_rate_df = pd.concat([zero_crossing_rate_df, zero_crossing_temp_df], ignore_index=True)
    spectral_bandwidth_df = pd.concat([spectral_bandwidth_df, spectral_bandwidth_temp_df], ignore_index=True)
    spectral_rolloff_df = pd.concat([spectral_rolloff_df, spectral_rolloff_temp_df], ignore_index=True)

    # Merge all feature DataFrames on 'file_index'
    merged_df = (mfccs_df
                .merge(chroma_stft_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_contrast_df, left_index=True, right_index=True, how='inner')
                .merge(zero_crossing_rate_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_bandwidth_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_rolloff_df, left_index=True, right_index=True, how='inner'))

# Import the label (Fake or Real)
csv_path=os.path.join(absolute_path, "meta.csv")
mapping_file=pd.read_csv(csv_path)
mapping_file['file_index']=mapping_file.file.str.replace('.wav', '', regex=False).astype(int)
mapping_file=mapping_file.drop(columns='file')
print(mapping_file)

# Merge to retrieve label
master_audio_df_all=merged_df.merge(mapping_file, on='file_index',how='inner')

# Print final DataFrame
print(master_audio_df_all)

# Export as Parquet file
master_audio_df_all.to_parquet('raw_data/master_audio_df_balanced_all.parquet', engine='pyarrow')
