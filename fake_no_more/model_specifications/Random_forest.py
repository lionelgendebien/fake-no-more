from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fake_no_more.data_preprocessing import process_data
import pandas as pd

def load_and_preprocess_audio(csv_path, audio_folder, sample_rate=16000, duration=5, max_audio_length=40):
    """
    Load and preprocess audio files, returning Mel spectrograms as numpy arrays.
    """
    max_len = sample_rate * duration
    mapping_file = pd.read_csv(csv_path)
    filter_1 = mapping_file[mapping_file['audio_length'] >= 5]
    filter_2 = filter_1.sort_values(by=['speaker', 'label', 'audio_length'], ascending=[True, True, True])
    filter_2['cum_audio_length'] = filter_2.groupby(['speaker', 'label'])['audio_length'].cumsum()
    filter_2 = filter_2[filter_2['cum_audio_length'] <= max_audio_length]
    final_filtering = filter_2[['file_index', 'label']]

    mel_spectrograms = []
    labels = []
    label_encoder = LabelEncoder()

    for file_num in final_filtering.file_index:
        audio_file = os.path.join(audio_folder, f'{file_num}.wav')

        if os.path.exists(audio_file):
            y, _ = librosa.load(audio_file, sr=sample_rate)
            if len(y) >= max_len:
                middle_start = int((len(y) - sample_rate * duration) / 2)
                y = y[middle_start:middle_start + sample_rate * duration]

                mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
                mel_spectrograms.append(mel_spectrogram)

                label = final_filtering[final_filtering['file_index'] == file_num]['label'].values[0]
                labels.append(label)

    X = np.array(mel_spectrograms)
    max_time_steps = max([spec.shape[1] for spec in mel_spectrograms])
    X_padded = []

    for spec in mel_spectrograms:
        pad_width = max_time_steps - spec.shape[1]
        if pad_width > 0:
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        X_padded.append(spec)

    X = np.array(X_padded)
    X = X[..., np.newaxis]
    y = np.array(labels)
    y = label_encoder.fit_transform(y)

    return X, y


def random_model(X_train, X_val, X_test, y_train, y_val, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_val_pred = rf_classifier.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_class_report = classification_report(y_val, y_val_pred)

    y_test_pred = rf_classifier.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_class_report = classification_report(y_test, y_test_pred)

    return val_accuracy, val_class_report, test_accuracy, test_class_report


def run_pipeline(df):
    X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val = process_data(df)
    val_accuracy, val_class_report, test_accuracy, test_class_report = random_model(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    )

    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Classification Report:\n{val_class_report}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Classification Report:\n{test_class_report}")


df = pd.read_parquet('raw_data/master_audio_df_3000_all.parquet')
run_pipeline(df)
