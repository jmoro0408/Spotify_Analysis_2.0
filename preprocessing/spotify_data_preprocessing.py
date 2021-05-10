# Imports
import pandas as pd
from pathlib import Path

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

streams = pd.json_normalize(streams) #This is a really handy way of converting dict keys to column names

del spotify_data
del streams_list

class CleanDataFrame():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_unknowns(self):
        dataframe = self.dataframe
        dataframe_clean = dataframe[~dataframe["artistName"].str.contains("Unknown Artist")]
        dataframe_clean = dataframe[~dataframe["trackName"].str.contains("Unknown Track")]
        return dataframe_clean

clean_data = CleanDataFrame(streams)

streams = clean_data.remove_unknowns()




#streams = streams[~streams["artistName"].str.contains("Unknown Artist")]










