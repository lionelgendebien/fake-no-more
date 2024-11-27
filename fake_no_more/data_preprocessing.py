from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Create X and y
def create_features_and_target(df):
    required_columns = {'file_index', 'speaker', 'label'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    X = df.drop(columns=['file_index', 'speaker', 'label'])
    y = df['label']
    return X, y

# Label encoding
def y_encoding(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y.values.ravel())

# Train-test split
def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    return X_train, X_test, y_train, y_test

# Train-val split
def split_train_val(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20)
    return X_train, X_val, y_train, y_val

# Feature scaling
def min_max_scaler(X_train, X_test, X_val):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_test_scaled, X_val_scaled

# Preprocess data
def process_data(df):
    X, y =create_features_and_target(df)
    y = y_encoding(y)
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    X_train, X_val, y_train, y_val = split_train_val(X_train,y_train)
    X_train_scaled, X_test_scaled, X_val_scaled = min_max_scaler(X_train, X_test, X_val)
    return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val
