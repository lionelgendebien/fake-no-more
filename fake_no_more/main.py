import pandas as pd
import os
from data_preprocessing import process_data, create_features_and_target, y_encoding, split_train_test, min_max_scaler

# Import master df
df=pd.read_parquet('raw_data/master_audio_df_3000_all.parquet', engine='pyarrow')

# Preprocess data
X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val=process_data(df)
