# Imports
import pandas as pd
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import time
import random
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from my_data_preprocessing import save_dataframe

SPOTIPY_CLIENT_ID = os.environ.get("client_id")
SPOTIPY_CLIENT_SECRET = os.environ.get("client_secret")
STREAMS_PICKLE_FILE = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\2. PreProcessing\PreProcessing_MyData\listening_history.pkl"
SAVE_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\2. PreProcessing\PreProcessing_MyData"


def build_df(dataframe):
    """
    Adds a track ID colume to dataframe for later use

    Parameters:
    Dataframe (pandas dataframe): Dataframe to have column added

    Returns:
    Dataframe (pandas dataframe): Original dataframe with new "trackID" column, filled with NaNs.
    """
    dataframe["trackId"] = np.nan
    return dataframe


def get_track_id(artist, track):
    """
    Returns spotify ID for a given track by specific artist.

    Parameters:
    artist (String): artist to whom track belongs to
    track (string): track to search for

    Returns:
    track_id (string): Spotify ID for the given track
    """
    track_id_dict = sp.search(q="artist:" + artist + " track:" + track, type="track")
    try:
        track_id = track_id_dict["tracks"]["items"][0]["id"]
    except IndexError:
        track_id = np.nan
    return track_id


def assign_ids(dataframe):
    """
    Uses the get_track_id function to assign IDs to all tracks in the givenm dataframe. 

    Parameters:
    dataframe (pandas dataframe): Dataframe to assigned track IDs to. Dataframe must have "trackID", "artistName" and "trackName" columns. 

    returns:
    dataframe (pandas dataframe): Original dataframe with full track IDs column. 
    """
    start = time.time()
    print("Getting song ids..")
    dataframe["trackId"] = dataframe.progress_apply(
        lambda x: get_track_id(x["artistName"], x["trackName"]), axis=1,
    )
    end = time.time()
    print(f"id's complete: {round((end - start),3)} seconds")
    return dataframe


def get_features(id):
    """
    Gathers audio features from Spotify for a given track ID. 
    See: https://developer.spotify.com/documentation/web-api/reference/#category-tracks for information on features

    Parameters:
    id (string): spotify ID corresponding to a given track

    returns:
    sp.audio_features([id]) (list): list containing dictionary of feature names (keys) and feature values (values.)
    """
    try:
        return sp.audio_features([id])
    except AttributeError:
        return np.nan


def grab_features(dataframe):
    """
    Attempts to assign song features using the get_features function to all songs in given dataframe. 
    This function creates a column that encompasses all features retuerned from Spotify in a json format for each track ID. 
    It then explodes this column into a seperate dataframe and concatenates it with the original.

    Parameters:
    dataframe (pandas dataframe): Dataframe to assigned track IDs to. Must have a "trackID" column

    Returns:
    dataframe (pandas dataframe): original pandas dataframe with song features included
    """
    start = time.time()
    print("Getting song features..")
    dataframe["features_json"] = dataframe["trackId"].progress_apply(
        get_features
    )  # progress apply allows for tqdm progress bar
    dataframe.dropna(
        axis=0, subset=["trackId"], inplace=True
    )  # cannot search for tracks that have no ID
    temp_list = [pd.json_normalize(x) for x in dataframe["features_json"]]
    features_df = pd.concat(x for x in temp_list).reset_index().drop(["index"], axis=1)
    dataframe = dataframe.reset_index().drop(["index"], axis=1)
    dataframe = pd.concat([dataframe, features_df], axis=1)
    dataframe.drop(["features_json"], axis=1, inplace=True)
    index_check = random.randint(
        0, len(dataframe)
    )  # performing check that temporary song feature df matches orignal df
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
    load_dotenv(find_dotenv())
    tqdm.pandas()  # required to use tqdm progress bar with pandas .apply
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(),
        requests_timeout=10,
        retries=10,
    )
    streams_total = pd.read_pickle(STREAMS_PICKLE_FILE)
    streams_total = streams_total[0:10]  # only getting the first 10 songs for testing
    streams_total = build_df(streams_total)
    streams_total = assign_ids(streams_total)
    streams_total = grab_features(streams_total)

    save_dataframe(streams_total, SAVE_DIR, "my_songs_features")
