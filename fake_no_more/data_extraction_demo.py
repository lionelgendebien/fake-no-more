import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler

# 'audio_test' argument takes the form of the full path of the wav test file
def prepare_test_data(audio_test):

    # Create a DataFrame for each feature
    mfccs_df = pd.DataFrame()
    chroma_stft_df = pd.DataFrame()
    spectral_contrast_df = pd.DataFrame()
    zero_crossing_rate_df = pd.DataFrame()
    spectral_bandwidth_df = pd.DataFrame()
    spectral_rolloff_df = pd.DataFrame()

    # Set up sample rate for extraction
    SAMPLE_RATE = 16000

    # Extract y (audio wave)
    y, _ = librosa.load(audio_test, sr=SAMPLE_RATE)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_bands=6)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)

    # Create DataFrames for each feature
    mfccs_temp_df = pd.DataFrame(mfccs.T, columns=[f'mfcc_{i+1}' for i in range(mfccs.shape[0])])
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

    # Merge all feature DataFrames on 'index'
    X_test = (mfccs_df
                .merge(chroma_stft_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_contrast_df, left_index=True, right_index=True, how='inner')
                .merge(zero_crossing_rate_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_bandwidth_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_rolloff_df, left_index=True, right_index=True, how='inner'))

    # Apply min-max scaling
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Return df
    return X_test_scaled
