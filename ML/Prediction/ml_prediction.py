import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

RS500_FEATURES_PATH = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis\PreProcessing\PreProcessing_RS500\RS500_features.pkl"
TRAINED_MODEL_PATH = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis\ML\Training\trained_model.h5"
SAVE_DIR = r"C:\Users\JM070903\OneDrive - Jacobs\Documents\Python\Spotify Listening Analysis\Spotify_Analysis\ML\Prediction"


def clean_songs_df(  # TODO Fix relative importing so I dont have to copy this here
    input_dataframe: pd.DataFrame,
    columns_to_drop: list = None,
    artists_tracks_to_remove: dict = None,
    max_playcount: int = None,
) -> pd.DataFrame:
    """cleans the input dataframe ready for ML preprocessing. The fucntion removes columns, removes specific tracks by specific artists, and can apply capping to song plays. 
    The song play capping replaces all instances with songs above a specific max_playcount with that max_playcount value. 

    Args:
        input_dataframe (pd.DataFrame): dataframe object to be cleaned
        columns_to_drop (list, optional): columns to be removed. Defaults to None.
        artists_tracks_to_remove (dict, optional): dictionary of artists as keys and songs as values to be removed. Defaults to None.
        max_playcount (int, optional): value to cap playcount at. Defaults to None.

    Returns:
       cleaned_dataframe (pd.DataFrame): cleaned dataframe
    """
    cleaned_dataframe = input_dataframe.copy()  # don't want to edit the original df
    if columns_to_drop is not None:
        cleaned_dataframe.drop(columns_to_drop, inplace=True, axis=1)

    if artists_tracks_to_remove is not None:
        for key, value in artists_tracks_to_remove.items():
            cleaned_dataframe = cleaned_dataframe[
                (cleaned_dataframe["artistName"] != key)
                & (cleaned_dataframe["trackName"] != value)
            ]

    if max_playcount is not None:
        cleaned_dataframe["playCount"].where(
            cleaned_dataframe["playCount"] <= max_playcount, max_playcount, inplace=True
        )
    return cleaned_dataframe


def convert_duration(dataframe: pd.DataFrame) -> pd.DataFrame:
    """creates new column for song duration in seconds, converted from duration in milliseconds

    Args:
        dataframe (pandas dataframe): dataframe to have duration column added, requires "duration_ms" column

    Returns:
        pandas dataframe: original dataframe with additional duration column
    """
    dataframe["duration"] = dataframe["duration_ms"].divide(60000)
    dataframe.drop("duration_ms", axis=1, inplace=True)
    return dataframe


def scale_features(dataframe: pd.DataFrame) -> np.ndarray:
    """Scales input dataframe with standard scalar

    Args:
        dataframe (pd.DataFrame): input dataframe to be scaled]

    Returns:
        X (np.ndarray): scaled dataframe in numpy array format
    """

    X = dataframe.copy()
    standard_scalar = StandardScaler()
    X = standard_scalar.fit_transform(X)
    return X


def predict_songs(X: np.ndarray) -> pd.DataFrame:
    """created a dataframe holding the artist, trackname,. and relevent predicted play time for each trtack in the prediction dataframe

    Args:
        X (np.ndarray): X features to be used for prediction

    Returns:
        pd.DataFrame: Dataframe with artists, tracks, and predicted playcount
    """
    predictions = {
        "artist": [],
        "track": [],
        "predicted_playcount": [],
    }  # dict is quicker for looping that dataframe object
    for count, track in enumerate(
        X
    ):  # this loops through the entire X array and appending the artist, track, and predicted playcount to the dict
        predictions["artist"].append(
            RS500_features["artistName"].iloc[count]
        )  # get the respective artist for this X array index and append to dict
        predictions["track"].append(
            RS500_features["trackName"].iloc[count]
        )  # same as above for track name
        track = X[count].reshape(
            1, 12
        )  # reshaping into appropriate shape for keras model prediction
        predictions["predicted_playcount"].append(
            model.predict(track)
        )  # making predictions based on X array features
        if count % 100 == 0:
            print(f"{count} of {len(X)}")  # some output to monitor progress

    predictions_df = pd.DataFrame(
        predictions
    )  # converting dict to dataframe for readability and analysis
    predictions_df["predicted_playcount"] = predictions_df[
        "predicted_playcount"
    ].astype(float)
    return predictions_df


if __name__ == "__main__":
    columns_to_remove = [
        "type",
        "id",
        "uri",
        "track_href",
        "analysis_url",
        "time_signature",
        "trackId",
        "artistName",
        "albumName",
        "trackName",
    ]
    RS500_features = pd.read_pickle(RS500_FEATURES_PATH)
    prediction_features = clean_songs_df(RS500_features, columns_to_remove)
    prediction_features = convert_duration(prediction_features)
    model = keras.models.load_model(TRAINED_MODEL_PATH)
    X = scale_features(prediction_features)
    predictions = predict_songs(X)
    predictions.to_pickle(Path(SAVE_DIR, "predictions_dataframe" + ".pkl"))
