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


def get_track_id(artist, track):
    track_id_dict = sp.search(q="artist:" + artist + " track:" + track, type="track")
    try:
        track_id = track_id_dict["tracks"]["items"][0]["id"]
    except IndexError:
        track_id = "track id not available"
    return track_id


streams_total["trackId"] = np.nan
test_df = streams_total.copy()
test_df = test_df[60:85]


def assign_ids(dataframe=streams_total):
    count = 0
    for artist_name, track_name in zip(dataframe["artistName"], dataframe["trackName"]):
        dataframe["trackId"].iloc[count] = get_track_id(artist_name, track_name)
        print(f"gettids id's: {count} of {len(dataframe)} completed")
        count += 1
    return dataframe


if streams_total["trackId"].count() == 0:
    assign_ids(test_df)


def get_features(id):
    return sp.audio_features([id])


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
    streams_total[feature] = np.nan
    test_df[feature] = np.nan


failed_songs = {}


def assign_features(
    dataframe=streams_total, features=features_list, failed_song_dict=failed_songs
):
    count = 0

    for id_ in tqdm(dataframe["trackId"]):
        try:
            for feature in features:
                dataframe[feature].iloc[count] = get_features(id_)[0][feature]
            # print(f"getting features: {count} of {len(dataframe)} completed")
            count += 1
        except TypeError:
            for feature in features_list:
                dataframe[feature].iloc[count] = np.nan
            failed_song_dict["artistName"] = dataframe["artistName"].iloc[count]
            failed_song_dict["trackName"] = dataframe["trackName"].iloc[count]
            count += 1
    print(f"{len(failed_song_dict)-1} songs failed")


assign_features(test_df)

print(failed_songs)

current_directory = Path(__file__).resolve().parent
pickle_name = "streams_total_features_pickle.pkl"
test_df.to_pickle(Path(current_directory / pickle_name))