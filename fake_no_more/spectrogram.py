import pandas as pd
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Constants
SAMPLE_RATE = 16000
DURATION = 5  # 5 seconds
MAX_LEN = SAMPLE_RATE * DURATION
N_MFCC = 20  # Number of MFCCs
N_CHROMA = 12  # Chroma features
N_BANDS = 6  # Spectral contrast bands
N_TIME_FRAMES = 128  # Target time frames for all features
TARGET_FREQ_BINS = 128  # Set a target frequency bins size for consistency

# Load data mapping file
csv_path = "raw_data/data_mapping.csv"
mapping_file = pd.read_csv(csv_path)

# Filter 1: Augment each individual audio of < 5'' to 5''
mapping_file['audio_length_adjusted'] = mapping_file['audio_length'].apply(lambda x: 5 if x < 5 else x)

# Filter 2: Limit audio files to cumulative sum of 200 seconds after adjusting each audio to 5''
mapping_file['audio_length_adjusted'] = mapping_file['audio_length_adjusted'].apply(lambda x: 5 if x > 5 else x)
mapping_file = mapping_file.sort_values(by=['speaker', 'label', 'audio_length'], ascending=[True, True, False])
mapping_file['cum_audio_length_adjusted'] = mapping_file.groupby(['speaker', 'label'])['audio_length_adjusted'].cumsum()
mapping_file_filtered = mapping_file[mapping_file['cum_audio_length_adjusted'] <= 200]
final_filtering = mapping_file_filtered[['file_index', 'label']]

# Prepare lists to store features and labels
mel_spectrograms = []
mfccs_list = []
chroma_stft_list = []
spectral_contrast_list = []
labels = []

# Extract features from audio files
for file_num, label in zip(final_filtering['file_index'], final_filtering['label']):
    # Load audio file
    audio_file = os.path.join("raw_data/release_in_the_wild/", f'{file_num}.wav')
    y, _ = librosa.load(audio_file, sr=SAMPLE_RATE)

    if len(y) < MAX_LEN:  # If the audio is shorter than 5 seconds
        repeats = MAX_LEN // len(y) + 1  # Calculate the number of repeats needed
        y = np.tile(y, repeats)[:MAX_LEN]  # Repeat and trim to exactly 5 seconds
    else:
        start_idx = (len(y) - MAX_LEN) // 2  # Calculate starting index for mid-point extraction
        y = y[start_idx:start_idx + MAX_LEN]  # Extract the middle 5 seconds

    # Extract Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=N_TIME_FRAMES)  # Pad/truncate to the same length
    # Pad or truncate to TARGET_FREQ_BINS (frequency bins)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=TARGET_FREQ_BINS, axis=0)  # Fix number of frequency bins
    mel_spectrograms.append(mel_spectrogram)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfccs = librosa.util.fix_length(mfccs, size=N_TIME_FRAMES)  # Ensure same length
    # Pad or truncate to TARGET_FREQ_BINS (frequency bins)
    mfccs = librosa.util.fix_length(mfccs, size=TARGET_FREQ_BINS, axis=0)  # Fix number of frequency bins
    mfccs_list.append(mfccs)

    # Extract Chroma feature
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=N_CHROMA)
    chroma_stft = librosa.util.fix_length(chroma_stft, size=N_TIME_FRAMES)  # Ensure same length
    # Pad or truncate to TARGET_FREQ_BINS (frequency bins)
    chroma_stft = librosa.util.fix_length(chroma_stft, size=TARGET_FREQ_BINS, axis=0)  # Fix number of frequency bins
    chroma_stft_list.append(chroma_stft)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_bands=N_BANDS)
    spectral_contrast = librosa.util.fix_length(spectral_contrast, size=N_TIME_FRAMES)  # Ensure same length
    # Pad or truncate to TARGET_FREQ_BINS (frequency bins)
    spectral_contrast = librosa.util.fix_length(spectral_contrast, size=TARGET_FREQ_BINS, axis=0)  # Fix number of frequency bins
    spectral_contrast_list.append(spectral_contrast)

    # Store the label for the current file
    labels.append(label)

# Convert lists of features to numpy arrays
X_mel = np.array(mel_spectrograms)  # Shape will be (num_samples, time_frames, freq_bins)
X_mfccs = np.array(mfccs_list)  # Shape will be (num_samples, time_frames, n_mfcc)
X_chroma = np.array(chroma_stft_list)  # Shape will be (num_samples, time_frames, n_chroma)
X_spectral_contrast = np.array(spectral_contrast_list)  # Shape will be (num_samples, time_frames, n_bands)

# Normalize all features (scaling to range [0, 1])
X_mel = X_mel / np.max(X_mel)
X_mfccs = X_mfccs / np.max(X_mfccs)
X_chroma = X_chroma / np.max(X_chroma)
X_spectral_contrast = X_spectral_contrast / np.max(X_spectral_contrast)

# Stack features along the channel axis to create a multi-channel input for the CNN
X = np.stack([X_mel, X_mfccs, X_chroma, X_spectral_contrast], axis=-1)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)  # One-hot encode labels

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Print the shape of the data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Initialize and train the CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def initialize_model(input_shape, num_classes, learning_rate=0.01):
    """
    Initialize the CNN model structure.
    """
    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output to feed into fully connected layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Initialize the model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (time_frames, num_features, num_channels)
num_classes = y_train.shape[1]
model = initialize_model(input_shape, num_classes)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
