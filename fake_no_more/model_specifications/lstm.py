from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from fake_no_more.data_preprocessing import process_data, process_data_LSTM


def reshape_x(X_train_scaled, X_val_scaled, X_test_scaled):
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    return X_train_reshaped, X_val_reshaped, X_test_reshaped

def initialize_model_LSTM():
    model_LSTM = Sequential()
    model_LSTM.add(Input(shape=(X_train_scaled[1],X_train_scaled[2])))

    # First LSTM layer
    model_LSTM.add(LSTM(64, return_sequences=True))
    model_LSTM.add(Dropout(0.2))

    # Second LSTM layer
    model_LSTM.add(LSTM(64, return_sequences=True))
    model_LSTM.add(Dropout(0.2))

    # Third LSTM layer
    model_LSTM.add(LSTM(64, return_sequences=True))
    model_LSTM.add(Dropout(0.2))

    # Fourth LSTM layer
    model_LSTM.add(LSTM(64, return_sequences=False))
    model_LSTM.add(Dropout(0.2))

    # First Dense layer
    model_LSTM.add(Dense(64, activation='relu'))
    model_LSTM.add(Dropout(0.2))

    # Second Dense layer
    model_LSTM.add(Dense(32, activation='relu'))
    model_LSTM.add(Dropout(0.2))

    # Output layer
    model_LSTM.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model_LSTM.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy', 'precision', 'recall'])
    return model_LSTM

def fit_LSTM(model):
    callback = EarlyStopping(patience= 5, restore_best_weights = True)
    model = initialize_model_LSTM()
    history = model.fit(X_train, y_train, epochs=50, callbacks = callback, validation_data=(X_val, y_val))
    return history

def evaluate_LSTM(model):
    score = model.evaluate(X_test, y_test)
    return score         

df=pd.read_parquet('raw_data/master_audio_df_balanced_all.parquet', engine='pyarrow')
X_train, X_test, X_val, y_train, y_test, y_val=process_data_LSTM(df)
print (X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
#X_train_reshaped, X_val_reshaped, X_test_reshaped = reshape_x(X_train_scaled, X_val_scaled, X_test_scaled)
breakpoint()
model = initialize_model_LSTM()
history = fit_LSTM(model)
score = evaluate_LSTM(model)
print(score)
print(plot_loss_accuracy(history, title=None))
