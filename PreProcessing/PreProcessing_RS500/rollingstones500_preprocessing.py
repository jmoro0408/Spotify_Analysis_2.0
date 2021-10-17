import pandas as pd
from pathlib import Path


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
    drop_columns = ["Number", "Genre", "Subgenre"]
    RS500_df = get_csv_data(RS500_DATA_PATH)
    RS500_df = clean_dataframe(RS500_df, drop_columns)
    save_dataframe(RS500_df, SAVE_DIR, "RS500")
