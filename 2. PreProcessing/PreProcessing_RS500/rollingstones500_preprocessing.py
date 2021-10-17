import pandas as pd
from pathlib import Path
from collections import Counter

rollingstones_data = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/data/raw_data/rollingstones500/rollingstones500.csv"

stones_df = pd.read_csv(rollingstones_data)
stones_df.drop(["Number", "Genre", "Subgenre"], axis=1, inplace=True)
current_directory = Path(__file__).resolve().parent
pickle_name = "stones_data.pkl"
stones_df.to_pickle(Path(current_directory / "pickles" / pickle_name))
