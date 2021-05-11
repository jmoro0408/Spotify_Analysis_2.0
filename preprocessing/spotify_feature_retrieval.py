"""
This script will retrieve song features from an inputted dataframe. 
SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required to be set as environment variables, or passed manually 
"""

# Imports
import pandas as pd
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from tqdm import tqdm

# Setting up and reading pickles
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
streams_pickle_file = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/streams_total_pickle.pkl"
streams_total = pd.read_pickle(streams_pickle_file)


streams_total = streams_total[:10]  # only getting the first 100 songs for now


def build_df(dataframe):
    dataframe["trackId"] = np.nan
    features_list = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    for feature in features_list:
        dataframe[feature] = np.nan

    return dataframe, features_list


def get_track_id(artist, track):
    track_id_dict = sp.search(q="artist:" + artist + " track:" + track, type="track")
    try:
        track_id = track_id_dict["tracks"]["items"][0]["id"]
    except IndexError:
        track_id = "track id not available"
    return track_id


def assign_ids(dataframe=streams_total):
    count = 0
    for artist_name, track_name in zip(dataframe["artistName"], dataframe["trackName"]):
        dataframe["trackId"].iloc[count] = get_track_id(artist_name, track_name)
        print(f"getting ids id's: {count+1} of {len(dataframe)} completed")
        count += 1
    return dataframe


def get_features(id):
    return sp.audio_features([id])


def assign_features(dataframe, features):
    count = 0
    failed_songs = {}
    print("Getting song features")
    for id_ in tqdm(dataframe["trackId"]):
        try:
            for feature in features:
                dataframe[feature].iloc[count] = get_features(id_)[0][feature]
            # print(f"getting features: {count} of {len(dataframe)} completed")
            count += 1
        except TypeError:
            for feature in features_list:
                dataframe[feature].iloc[count] = np.nan
            failed_songs["artistName"] = dataframe["artistName"].iloc[count]
            failed_songs["trackName"] = dataframe["trackName"].iloc[count]
            count += 1
    print(f"{len(failed_songs)-1} songs failed")
    return dataframe, failed_songs


if __name__ == "__main__":
    streams_total, features_list = build_df(streams_total)
    streams_total = assign_ids(streams_total)
    streams_total, failed_songs = assign_features(
        dataframe=streams_total, features=features_list
    )

    print(failed_songs)
    print(streams_total.head())

current_directory = Path(__file__).resolve().parent
pickle_name = "streams_total_features_pickle.pkl"
streams_total.to_pickle(Path(current_directory / pickle_name))