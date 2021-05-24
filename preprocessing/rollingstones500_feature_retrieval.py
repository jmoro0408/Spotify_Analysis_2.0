from numpy.lib.type_check import nan_to_num
import pandas as pd
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import time
import random
from tqdm import tqdm
from my_data_spotify_feature_retrieval import (
    get_track_id,
    assign_ids,
    get_features,
    grab_features,
)

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(),
    requests_timeout=50,
    retries=20,
)
rollingstones500_file = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/pickles/rollingstones_csv.pkl"
stones = pd.read_pickle(rollingstones500_file)


def get_album_id(artist, album):
    query = "artist:" + artist + " album:" + album
    album = sp.search(q=query, type="album")
    try:
        album_id = album["albums"]["items"][0]["uri"]
    except IndexError:
        album_id = None
    return album_id


def get_album_tracks(artist, album):
    """
    Should speed this up by using a .apply method...but there's only a total of 500 albums so it's not a huge time issue
    """

    album_id = get_album_id(artist, album)
    if album_id == None:
        pass
    else:
        album_songs = []
        tracks = sp.album_tracks(album_id)
        for track in tracks["items"]:
            album_songs.append(track["name"])
        album_dict = {
            "artistName": artist,
            "albumName": album,
            "trackName": album_songs,
        }
        return album_dict


def get_rollingstones_tracks(df):
    list_of_dict_tracks = []
    print("Getting album ids..")
    for album, artist in tqdm(zip(df["Album"], df["Artist"]), total=len(df)):
        current_album_tracks = get_album_tracks(artist, album)
        list_of_dict_tracks.append(current_album_tracks)
    list_of_dict_tracks = [x for x in list_of_dict_tracks if x is not None]
    return list_of_dict_tracks


def build_stones_df():
    song_df = pd.DataFrame(columns=["artistName", "albumName", "trackName"])
    for i in range(len(list_of_dict_tracks)):
        artist = list_of_dict_tracks[i]["artistName"]
        album = list_of_dict_tracks[i]["albumName"]
        tracks = list_of_dict_tracks[i]["trackName"]
        df = pd.DataFrame(
            [list_of_dict_tracks[i]], columns=list_of_dict_tracks[0].keys()
        )
        song_df = (
            pd.concat([song_df, df], axis=0)
            .reset_index()
            .explode(column="trackName", ignore_index=True)
            .drop(["index"], axis=1)
        )
    return song_df


# test_stones = stones.copy()[:15]
# test_album = stones["Album"].iloc[0]
# test_artist = stones["Artist"].iloc[0]
# test_tracks = get_album_tracks(test_artist, test_album)

if __name__ == "__main__":
    list_of_dict_tracks = get_rollingstones_tracks(stones)
    song_df = build_stones_df()
    song_df = assign_ids(song_df)
    song_df = grab_features(song_df)

print(song_df[:20])

current_directory = Path(__file__).resolve().parent
pickle_name = "rolling_stones_features.pkl"
# song_df.to_pickle(Path(current_directory / "pickles" / pickle_name))
