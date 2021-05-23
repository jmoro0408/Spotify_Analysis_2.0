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


tqdm.pandas()

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
rollingstones500_file = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/pickles/stones_data.pkl"
stones = pd.read_pickle(rollingstones500_file)


def get_album_id(artist, album):
    query = "artist:" + artist + " album:" + album
    album = sp.search(q=query, type="album")
    album_id = album["albums"]["items"][0]["uri"]
    return album_id


def get_album_tracks(artist, album):
    album_id = get_album_id(artist, album)
    album_songs = []
    tracks = sp.album_tracks(album_id)
    for track in tracks["items"]:
        album_songs.append(track["name"])
    album_dict = {"artistName": artist, "albumName": album, "trackName": album_songs}
    return album_dict


test_stones = stones.copy()[:5]
test_album = stones["Album"].iloc[0]
test_artist = stones["Artist"].iloc[0]

test_tracks = get_album_tracks(test_artist, test_album)

list_of_dict_tracks = []
for album, artist in zip(test_stones["Album"], test_stones["Artist"]):
    current_album_tracks = get_album_tracks(artist, album)
    list_of_dict_tracks.append(current_album_tracks)


song_df = pd.DataFrame(columns=["artistName", "albumName", "trackName"])
for i in range(len(list_of_dict_tracks)):
    artist = list_of_dict_tracks[i]["artistName"]
    album = list_of_dict_tracks[i]["albumName"]
    tracks = list_of_dict_tracks[i]["trackName"]
    df = pd.DataFrame([list_of_dict_tracks[i]], columns=list_of_dict_tracks[0].keys())
    song_df = (
        pd.concat([song_df, df], axis=0)
        .reset_index()
        .explode(column="trackName", ignore_index=True)
        .drop(["index"], axis=1)
    )

song_df = assign_ids(song_df)
song_df = grab_features(song_df)

print(song_df.head())