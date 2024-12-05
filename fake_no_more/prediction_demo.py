import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime

try:
    print("Avoiding lazy import", datetime.now())
    librosa.load("some_file.wav")
except:
    print("Nothing bad happened.just trying", datetime.now())

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
    DURATION = 5
    MAX_LEN = SAMPLE_RATE * DURATION
    print("before librosa", datetime.now())
    # Extract y (audio wave)
    y, _ = librosa.load(audio_test, sr=SAMPLE_RATE)
    print("after librosa", datetime.now())

    # Restrict to first 5''
    y = y[:MAX_LEN]

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_bands=6)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)

    print("after features", datetime.now())
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

    print("before merge", datetime.now())
    # Merge all feature DataFrames on 'index'
    X_test = (mfccs_df
                .merge(chroma_stft_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_contrast_df, left_index=True, right_index=True, how='inner')
                .merge(zero_crossing_rate_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_bandwidth_df, left_index=True, right_index=True, how='inner')
                .merge(spectral_rolloff_df, left_index=True, right_index=True, how='inner'))

    print("before minmax", datetime.now())
    # Apply min-max scaling
    scaler = MinMaxScaler()
    print("before fit", datetime.now())
    X_test_scaled = scaler.fit_transform(X_test)
    print("before fit", datetime.now())

    # Reshape
    time_steps = 157  # Number of rows per audio file
    num_audio = len(X_test) // time_steps
    features = X_test.shape[1]# Number of features per row
    X_reshaped = X_test.reshape(num_audio, time_steps, features)

    # Return df
    return X_reshaped

def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Use the loaded model
def predict_X(loaded_model, X_test):
    predictions = loaded_model.predict(X_test)
    if predictions>0.5:
        return 'FAKE'
    elif predictions<0.5:
        return 'REAL'
    else:
        return 'UNCLEAR'

# Predict probability
def predict_prob(loaded_model, X_test):
    predictions = loaded_model.predict(X_test)
    return predictions
