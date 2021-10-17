# Imports
import pandas as pd
from pathlib import Path
from collections import Counter

SPOTIFY_DATA_FOLDER_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\1. Data\MyData"


def gather_mydata(folder_dir):
    """
    Returns consolidated dataframe of listening history from Spotify .json files

    Parameters:
    folder_dir (string): folder directory location of .json Spotify files
    
    Returns:

    spotify_data_df (pandas dataframe): Dataframe object containing combined and consolidated listening history
    """

    _folder_dir = Path(folder_dir)  # converting to pathlib object for OS-agnostic manipulation
    _jsons = Path(_folder_dir).glob("*.json")  # Finding all files in the json folder that end with .json -> generator object
    _json_list = [file.name for file in _jsons]  # retrieving filename of .json files only
    _streaming_list = [s for s in _json_list if "StreamingHistory" in s] #grabbing .jsons for streaming history only
    _spotify_data = {key: [] for key in _streaming_list} # Creating empty dict with json filenames as keys

    for spotify_json in _streaming_list:
        json_filepath = (Path(_folder_dir, spotify_json))    
        read_data = pd.read_json(json_filepath, typ="series", encoding="utf8")
        _spotify_data[spotify_json].append(read_data)

    streams_list = [key[0] for key in _spotify_data.values()]


    spotify_data_df = pd.concat(streams_list, ignore_index=True, sort=False)
    spotify_data_df = pd.json_normalize(spotify_data_df)  # This is a really handy way of converting dict keys to column names
   
    return spotify_data_df
streams_df = gather_mydata(SPOTIFY_DATA_FOLDER_DIR)


"""

my_spotify_jsons = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/data/raw_data/my_spotify_jsons"
list_of_jsons = []
jsons = Path(my_spotify_jsons).glob(
    "*.json"
)  # Finding all files in the json folder that end with .json
for file in jsons:
    list_of_jsons.append(file.name)

spotify_data = {key: [] for key in list_of_jsons}

for spotify_json in list_of_jsons:
    json_filepath = Path(my_spotify_jsons + "/" + spotify_json)
    read_data = pd.read_json(json_filepath, typ="series", encoding="utf8")
    spotify_data[spotify_json].append(read_data)

streams_list = [
    spotify_data["StreamingHistory0.json"][0],
    spotify_data["StreamingHistory1.json"][0],
    spotify_data["StreamingHistory2.json"][0],
    spotify_data["StreamingHistory3.json"][0],
]

streams = pd.concat(streams_list, ignore_index=True, sort=False)

streams = pd.json_normalize(
    streams
)  # This is a really handy way of converting dict keys to column names

del spotify_data
del streams_list


class CleanDataFrame:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_track(self, tracknames):
        for track in tracknames:
            self.dataframe = self.dataframe[
                ~self.dataframe["trackName"].str.contains(track)
            ]
        return self.dataframe

    def remove_artist(self, artists):
        for artist in artists:
            self.dataframe = self.dataframe[
                ~self.dataframe["artistName"].str.contains(artist)
            ]
        return self.dataframe

    def remove_unwanted(self):
        tracks_to_remove = [
            "Binaural Beta Sinus Drone II",
            "Binaural Alpha Sinus 100 Hz - 108 Hz",
            "Cabin Sound",
            "Unknown Track",
        ]  # Most of these songs are white noise I listen to on repeat, need to remove them from the df
        self.remove_track(tracknames=tracks_to_remove)

        artists_to_remove = [
            "Unknown Artist",
            "Stuff You Should Know",
            "Freakonomics Radio",
            "World War One",
            "The History of WWII Podcast - by Ray Harris Jr",
        ]  # Removing podcasts and unknown tracks
        self.remove_artist(artists=artists_to_remove)

        return self

    def convert_datetime(self):
        self.dataframe["endTime"] = pd.to_datetime(
            self.dataframe["endTime"], format="%Y-%m-%d %H:%M"
        )
        # self.dataframe["Date"] = pd.to_datetime(self.dataframe["endTime"].dt.date)
        # self.dataframe["Time"] = self.dataframe["endTime"].dt.time
        self.dataframe["minutesPlayed"] = self.dataframe["msPlayed"].divide(60000)

        self.dataframe.drop(["endTime", "msPlayed"], axis=1, inplace=True)

        return self

    def return_df(self):
        # need to add this at the end to return the df instead of a CleanDataFrame object
        return self.dataframe


clean_data = CleanDataFrame(streams)
streams = clean_data.remove_unwanted().convert_datetime().return_df()

duplicate_count = Counter(list(streams["trackName"]))
# print(duplicate_count.most_common()[:5])

# I have many tracks that I have played more than once, and they are counted seperately each time. I want to combine these so I just have the total time played for each track
streams_total = streams.copy()
streams_total["minutesTotal"] = streams_total.groupby(by=["trackName"])[
    "minutesPlayed"
].transform("sum")
streams_total = streams_total.drop_duplicates(subset=["trackName"]).drop(
    ["minutesPlayed"], axis=1
)

current_directory = Path(__file__).resolve().parent
pickle_name = "my_streams_pickle.pkl"
streams_total.to_pickle(Path(current_directory / "pickles" / pickle_name))

"""