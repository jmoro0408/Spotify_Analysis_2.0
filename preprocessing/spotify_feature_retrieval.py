"""
This script will retrieve song features form spotify from an inputted dataframe. 
SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required to be set as environment variables, or passed manually 
"""

# Imports
import pandas as pd
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from tqdm import tqdm
import time
import random

# Setting up and reading pickles
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
streams_pickle_file = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/pickles/my_streams_pickle.pkl"
streams_total = pd.read_pickle(streams_pickle_file)
streams_total = streams_total[0:10]  # only getting the first 10 songs for testing


def build_df(dataframe):
    dataframe["trackId"] = np.nan
    return dataframe


def get_track_id(artist, track):
    track_id_dict = sp.search(q="artist:" + artist + " track:" + track, type="track")
    try:
        track_id = track_id_dict["tracks"]["items"][0]["id"]
    except IndexError:
        track_id = np.nan
    return track_id


def assign_ids(dataframe=streams_total):
    tqdm.pandas()
    print("Getting song ids..")
    dataframe["trackId"] = dataframe.progress_apply(
        lambda x: get_track_id(x["artistName"], x["trackName"]), axis=1
    )
    return dataframe


def get_features(id):
    try:
        return sp.audio_features([id])
    except AttributeError:
        return np.nan


def assign_features(dataframe, features):
    """
    This was my first attempt at assigning features, but looping through a dataframe is horribly slow. For 9000 songs this took almost 3 hrs.
    The .apply method below is 3x faster
    """
    start = time.time()
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
    # print(f"{len(failed_songs)-1} songs failed")
    end = time.time()
    print(f"for loop took {end - start} seconds")
    return dataframe, failed_songs


def grab_features(dataframe):
    tqdm.pandas()
    start = time.time()
    print("Getting song features..")
    dataframe["features_json"] = dataframe["trackId"].progress_apply(
        get_features
    )  # progress apply allows for tqdm progress bar
    dataframe.dropna(axis=0, subset=["trackId"], inplace=True)
    temp_list = [pd.json_normalize(x) for x in dataframe["features_json"]]
    features_df = pd.concat(x for x in temp_list).reset_index().drop(["index"], axis=1)
    dataframe = dataframe.reset_index().drop(["index"], axis=1)
    dataframe = pd.concat([dataframe, features_df], axis=1)
    dataframe.drop(["features_json"], axis=1, inplace=True)
    index_check = random.randint(0, len(dataframe))
    assert (
        dataframe["trackId"].iloc[index_check] == dataframe["id"].iloc[index_check]
    ), "track IDs do not match"
    del temp_list, features_df
    end = time.time()
    print(
        f".apply took {round((end - start),3)} seconds for {len(dataframe)} songs, around {round((end-start) / (len(dataframe)), 3)} seconds per song"
    )
    return dataframe


if __name__ == "__main__":
    streams_total = build_df(streams_total)
    streams_total = assign_ids(streams_total)
    streams_total = grab_features(streams_total)
    # print(streams_total.head())

current_directory = Path(__file__).resolve().parent
pickle_name = "my_features.pkl"
streams_total.to_pickle(Path(current_directory / "pickles" / pickle_name))