import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def convert_duration(dataframe):
    dataframe["duration"] = dataframe["duration_ms"].divide(60000)
    dataframe.drop("duration_ms", axis=1, inplace=True)
    return dataframe


stones_features = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/pickles/rolling_stones_features.pkl"
stones_features_raw = pd.read_pickle(stones_features)
stones_features = stones_features_raw.copy()

columns_to_drop = [
    "type",
    "id",
    "uri",
    "track_href",
    "analysis_url",
    "time_signature",
    "trackId",
    "artistName",
    "albumName",
    "trackName",
]
# dropping unhelpful features
stones_features.drop(columns_to_drop, inplace=True, axis=1)
stones_features = convert_duration(stones_features)

trained_model_dir = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/ml_stuff/trained_model.h5"
model = keras.models.load_model(trained_model_dir)
# model.summary()

X = stones_features.copy()
standard_scalar = StandardScaler()
X = standard_scalar.fit_transform(X)

predictions = {"artist": [], "track": [], "predicted_playcount": []}
for count, track in enumerate(X):
    predictions["artist"].append(stones_features_raw["artistName"].iloc[count])
    predictions["track"].append(stones_features_raw["trackName"].iloc[count])
    track = X[count].reshape(1, 12)
    predictions["predicted_playcount"].append(model.predict(track))
    if count % 100 == 0:
        print(f"{count} of {len(X)}")

predictions_df = pd.DataFrame(predictions)
predictions_df["predicted_playcount"] = predictions_df["predicted_playcount"].astype(
    float
)
current_directory = os.getcwd()
pickle_name = "prediction_dataframe.pkl"
predictions_df.to_pickle(os.path.join(current_directory, pickle_name))