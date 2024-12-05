from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import math
import pickle
import numpy as np

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

def reshaping_X_LSTM(X,y):
    time_steps = 157  # Number of rows per audio file
    num_audio = len(X) // time_steps
    features = X.shape[1]# Number of features per row
    X_reshaped = X.to_numpy().reshape(num_audio, time_steps, features)
    y_reshaped = y.reshape(-1, time_steps)[:, 0]
    return X_reshaped, y_reshaped


# Train-test split
def split_train_test_LSTM(X,X_reshaped, y_reshaped):
    time_steps = 157  # Number of rows per audio file
    num_audio = len(X) // time_steps
    test_size = math.ceil(num_audio * 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size = test_size, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

# Train-val split
def split_train_val_LSTM(X_train, y_train):
    test_size = math.floor(X_train.shape[0]*0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = test_size, random_state=42, shuffle=True)
    return X_train, X_val, y_train, y_val

# Feature scaling
# def min_max_scaler_LSTM(X_train, X_test, X_val):
#     scaler = MinMaxScaler()  # Create the scaler
#     scaler.fit(X_train)  # Fit the scaler on the training data
#     scalerfile = 'scaler_fitted.sav'
#     pickle.dump(scaler, open(scalerfile, 'wb'))  # Save the fitted scaler

#     # Transform the data
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     X_val_scaled = scaler.transform(X_val)

#     return X_train_scaled, X_test_scaled, X_val_scaled

# Preprocess data
# def process_data_LSTM(df):
#     X, y =create_features_and_target(df)
#     y = y_encoding(y)
#     X_reshaped, y_reshaped = reshaping_X_LSTM(X,y)
#     X_train, X_test, y_train, y_test = split_train_test_LSTM(X, X_reshaped,y_reshaped)
#     X_train, X_val, y_train, y_val = split_train_val_LSTM(X_train,y_train)
#     # X_train_scaled, X_test_scaled, X_val_scaled=min_max_scaler_LSTM(X_train, X_test, X_val)
#     return X_train, X_test, X_val, y_train,y_test, y_val
