# Imports
import pandas as pd
from pathlib import Path


SPOTIFY_DATA_FOLDER_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\Data\MyData"
SAVE_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\PreProcessing\PreProcessing_MyData"


class CleanDataFrame:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_track(self, tracknames):
        """
        Removes any tracks passed to argument
        
        Parameters:
        tracknames (list): list of tracks to remove
        
        Returns:
        self: original dataframe object with tracks removed
        
        """
        for track in tracknames:
            self.dataframe = self.dataframe[
                ~self.dataframe["trackName"].str.contains(track)
            ]
        return self

    def remove_artist(self, artists):
        """
        Removes any artists passed to argument
        
        Parameters:
        artists (list): list of artists to remove
        
        Returns:
        self: original dataframe object with artists removed
        
        """
        for artist in artists:
            self.dataframe = self.dataframe[
                ~self.dataframe["artistName"].str.contains(artist)
            ]
        return self

    def convert_datetime(self):
        """
        Converts all time based df columns to datetime objects
        
        Parameters:
        self (dataframe): original dataframe to be cleaned
        
        Returns:
        self (dataframe): cleaned dataframe
        """
        self.dataframe["endTime"] = pd.to_datetime(
            self.dataframe["endTime"], format="%Y-%m-%d %H:%M"
        )
        self.dataframe["minutesPlayed"] = self.dataframe["msPlayed"].divide(60000)

        self.dataframe.drop(["msPlayed"], axis=1, inplace=True)

        return self

    def remove_duplicates(self):
        """
        Combines the play time of duplicate tracks into a single total play time track
        
        Parameters:
        self (dataframe): original dataframe to have duplicates combined
        
        Returns:
        self (dataframe): cleaned dataframe with duplicate song data combined
        """
        self.dataframe["minutesTotal"] = self.dataframe.groupby(by=["trackName"])[
            "minutesPlayed"
        ].transform("sum")
        self.dataframe.drop_duplicates(subset=["trackName"]).drop(
            ["minutesPlayed"], axis=1
        )

        return self

    def return_df(
        self,
    ):  # ToDo - shouldn't need this method. The dataframe shoud be returned when any of the above methods are called.
        """ Returns the dataframe object, rather than a CleanDataFrame object. """
        return self.dataframe


def gather_mydata(folder_dir):
    """
    Returns consolidated dataframe of listening history from Spotify .json files

    Parameters:
    folder_dir (string): folder directory location of .json Spotify files
    
    Returns:

    spotify_data_df (pandas dataframe): Dataframe object containing combined and consolidated listening history
    """

    _folder_dir = Path(
        folder_dir
    )  # converting to pathlib object for OS-agnostic manipulation
    _jsons = Path(_folder_dir).glob(
        "*.json"
    )  # Finding all files in the json folder that end with .json -> generator object
    _json_list = [
        file.name for file in _jsons
    ]  # retrieving filename of .json files only
    _streaming_list = [
        s for s in _json_list if "StreamingHistory" in s
    ]  # grabbing .jsons for streaming history only
    _spotify_data = {
        key: [] for key in _streaming_list
    }  # Creating empty dict with json filenames as keys

    for spotify_json in _streaming_list:
        json_filepath = Path(_folder_dir, spotify_json)
        read_data = pd.read_json(json_filepath, typ="series", encoding="utf8")
        _spotify_data[spotify_json].append(read_data)

    streams_list = [key[0] for key in _spotify_data.values()]

    spotify_data_df = pd.concat(streams_list, ignore_index=True, sort=False)
    spotify_data_df = pd.json_normalize(
        spotify_data_df
    )  # This is a really handy way of converting dict keys to column names

    return spotify_data_df


def save_dataframe(cleaned_dataframe, save_directory, filename):
    """
    Saves the cleaned dataframe as a pickle file 

    Parameters:
    cleaned_dataframe (dataframe): pandas dataframe object to be saved
    save_directory (string): folder directory of where cleaned dataframe .pkl is to be saved
    filename (string): name of .pkl file to be saved

    Returns:
    saved_pickle (.pkl file): .pkl file of dataframe, saved in user submitted save directory, with chosen filename
    """

    _save_directory = Path(save_directory, filename + ".pkl")
    return cleaned_dataframe.to_pickle(_save_directory)


if __name__ == "__main__":
    tracks_to_remove = [
        "Binaural Beta Sinus Drone II",
        "Binaural Alpha Sinus 100 Hz - 108 Hz",
        "Cabin Sound",
        "Unknown Track",
        "White Noise - 200 hz"
    ]  # Most of these songs are white noise I listen to on repeat, need to remove them from the df
    artists_to_remove = [
        "Unknown Artist",
        "Stuff You Should Know",
        "Freakonomics Radio",
        "World War One",
        "The History of WWII Podcast - by Ray Harris Jr",
    ]  # Removing podcasts and unknown tracks

    streams_df = gather_mydata(SPOTIFY_DATA_FOLDER_DIR)
    clean_data = CleanDataFrame(streams_df)
    streams_df = (
        clean_data.remove_track(tracks_to_remove)
        .remove_artist(artists_to_remove)
        .convert_datetime()
        .remove_duplicates()
        .return_df()
    )
    save_dataframe(streams_df, SAVE_DIR, "listening_history")
