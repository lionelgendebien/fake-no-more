from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from fake_no_more.data_preprocessing import process_data, process_data_LSTM

def initialize_model_LSTM(X_train_scaled):
    model_LSTM = Sequential()
    model_LSTM.add(Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))


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

    optimizer = Adam(learning_rate=0.0001)
    model_LSTM.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy', 'precision', 'recall'])
    return model_LSTM

def fit_LSTM(X_train_scaled, model):
    callback = EarlyStopping(patience= 15, restore_best_weights = True)
    model.fit(X_train_scaled, y_train, epochs=50, callbacks = callback, validation_data=(X_val_scaled, y_val))
    return model

def evaluate_LSTM(model):
    score = model.evaluate(X_test_scaled, y_test)
    return score

df=pd.read_parquet('raw_data/master_audio_df_balanced_all.parquet', engine='pyarrow')
X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val=process_data_LSTM(df)
print (X_train_scaled.shape, X_test_scaled.shape, X_val_scaled.shape, y_train.shape, y_test.shape, y_val.shape)
model = initialize_model_LSTM(X_train_scaled)
model = fit_LSTM(X_train_scaled, model)
score = evaluate_LSTM(model)
print(score)

# Save the model
with open('lstm.pkl', 'wb') as file:
    pickle.dump(model, file)

# # Load the model
# with open('lstm.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# # Use the loaded model
# predictions = loaded_model.predict(X)
# print(predictions)
