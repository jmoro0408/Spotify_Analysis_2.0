import pandas as pd
from tqdm import tqdm
from ...PreProcessing.PreProcessing_MyData.my_data_preprocessing import save_dataframe
from ...PreProcessing.PreProcessing_MyData.my_data_spotify_feature_retrieval import assign_ids, grab_features, sp


RS500_PKL = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\PreProcessing\PreProcessing_RS500\RS500.pkl"
SAVE_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\PreProcessing\PreProcessing_RS500"


def get_album_id(artist, album):
    """
    Returns the spotify album ID for a given album and artist

    Parameters:
    artist (string): Artist to whom the album belongs to
    album (String): name of the album to return ID for

    Returns:
    album_id (string): Spotify ID of the album
    """
    query = "artist:" + artist + " album:" + album
    album = sp.search(q=query, type="album")
    try:
        album_id = album["albums"]["items"][0]["uri"]
    except IndexError:
        album_id = None
    return album_id


def get_album_tracks(artist, album):  # TODO: Vectorize this function
    """
    Creates a dictionary of all the tracks on any album with the album title as the dict key
    Does this by first finding the album id using get_album_id function 

    Parameters:
    artist (string): Artist to whom the album belongs to
    album (String): name of the album to return tracks for

    Returns:
    album_dict (dict): Dictionary containing album name and tracks
    """

    album_id = get_album_id(artist, album)
    if album_id is None:
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
    """
    applies get_album_tracks function to RS500 df, creating a list of dictionaries with 
    key: album, value: [tracks]
    
    Parameters:
    df (pandas dataframe): Dartaframe to grab tracks of, should contain "Artist" and "Album" columns

    Return:
    list_of_dict_tracks (list): list of dictionaries with key: album, value: [tracks]
    """
    print("Getting album ids..")

    list_of_dict_tracks = []
    for album, artist in tqdm(zip(df["Album"], df["Artist"]), total=len(df)):
        current_album_tracks = get_album_tracks(artist, album)
        list_of_dict_tracks.append(current_album_tracks)
    list_of_dict_tracks = [x for x in list_of_dict_tracks if x is not None]
    return list_of_dict_tracks


def build_stones_df():
    """
    Create a pandas dataframe from the get_rollingstones_tracks function. 
    The df has the correct struture to begin grabbing track IDs and track features

    Parameters:
    None

    Returns:
    song_df (pandas dataframe): dataframe containing RS500 tracks
    """
    song_df = pd.DataFrame(columns=["artistName", "albumName", "trackName"])
    for i in range(len(list_of_dict_tracks)):
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


if __name__ == "__main__":
    print(
        "This file will retrieve all song features, this may take a while. Do you want to continue? [Y/N]"
    )
    user_check = input()
    if user_check.upper() == "Y":
        tqdm.pandas()  # required to use tqdm progress bar with pandas .apply
        stones = pd.read_pickle(RS500_PKL)
        list_of_dict_tracks = get_rollingstones_tracks(stones)
        RS500_song_df = build_stones_df()
        RS500_song_df = assign_ids(RS500_song_df)
        RS500_song_df = grab_features(RS500_song_df)
        save_dataframe(RS500_song_df, SAVE_DIR, "RS500_features")
    else:
        pass
