import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


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

X = stones_features.copy()
x_test = X.iloc[0].to_numpy()

trained_model_dir = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/ml_stuff/trained_model.h5"
model = keras.models.load_model(trained_model_dir)

print(model.predict(x_test))
