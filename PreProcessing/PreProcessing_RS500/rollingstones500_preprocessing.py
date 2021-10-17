import sys

sys.path.insert(
    0,
    r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\PreProcessing\PreProcessing_MyData",
)

import pandas as pd
from pathlib import Path
from my_data_preprocessing import save_dataframe


RS500_DATA_PATH = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\Data\rollingstones500.csv"
SAVE_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis_2.0\PreProcessing\PreProcessing_RS500"


def get_csv_data(CSV_path):
    """
    Imports CSV to pandas dataframe

    Parameters:
    CSV_path (string): full filepath of .csv file

    Returns:
    RS500_df (pandas dataframe): dataframe object containing .csv data
    """
    RS500_path = Path(CSV_path)
    RS500_df = pd.read_csv(RS500_path, encoding="latin-1")
    return RS500_df


def clean_dataframe(pandas_dataframe, columns_to_drop):
    clean_df = pandas_dataframe.drop(columns_to_drop, axis=1)
    return clean_df


if __name__ == "__main__":
    drop_columns = ["Number", "Genre", "Subgenre"]
    RS500_df = get_csv_data(RS500_DATA_PATH)
    RS500_df = clean_dataframe(RS500_df, drop_columns)
    save_dataframe(RS500_df, SAVE_DIR, "RS500")
