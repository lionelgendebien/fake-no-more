from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fake_no_more.data_preprocessing import process_data
import pandas as pd

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
