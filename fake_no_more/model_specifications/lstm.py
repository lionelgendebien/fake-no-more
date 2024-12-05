from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pickle

from fake_no_more.data_preprocessing import process_data_LSTM

def initialize_model_LSTM(X_train):
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model_LSTM = Sequential()

    model_LSTM.add(normalizer)

    # First LSTM layer
    model_LSTM.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
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

def fit_LSTM(X_train, model):
    callback = EarlyStopping(patience= 15, restore_best_weights = True)
    model.fit(X_train, y_train, epochs=50, callbacks = callback, validation_data=(X_val, y_val))
    return model

def evaluate_LSTM(model):
    score = model.evaluate(X_test, y_test)
    return score

df=pd.read_parquet('raw_data/master_audio_df_balanced_all.parquet', engine='pyarrow')
X_train, X_test, X_val, y_train, y_test, y_val=process_data_LSTM(df)
model = initialize_model_LSTM(X_train)
model = fit_LSTM(X_train, model)
score = evaluate_LSTM(model)
print(score)

# Save the model
with open('lstm_new.pkl', 'wb') as file:
    pickle.dump(model, file)

# # Load the model
# with open('lstm_new.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# # Use the loaded model
# predictions = loaded_model.predict(X)
# print(predictions)
