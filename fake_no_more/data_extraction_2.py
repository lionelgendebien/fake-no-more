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

    # Filter 1: keep audio files of min 5''
filter_1=mapping_file[mapping_file['audio_length'] >= 5]

    # Filter 2:keep audio files whose total cumul length per speaker per label
    # (i.e. Fake or Real) does not exceed 40'' (NB: only 7 out of 54 speakers
    # have total audio length of either fake or real below 40'')
filter_2=filter_1.sort_values(by=['speaker', 'label', 'audio_length'], ascending=[True, True, True])
filter_2['cum_audio_length'] = filter_2.groupby(['speaker', 'label'])['audio_length'].cumsum()
filter_2 = filter_2[filter_2['cum_audio_length'] <= 40]
final_filtering=filter_2[['file_index', 'label']]

for file_num in final_filtering.file_index:
    # Load audio file
    audio_file = os.path.join(absolute_path, f'{file_num}.wav')
    y, _ = librosa.load(audio_file, sr=SAMPLE_RATE)

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
