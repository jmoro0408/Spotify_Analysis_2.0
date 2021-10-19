"""
This uses the ml_exploration_cap jupyter notebook as a basis for creating a neural net 
to make predictions on how many time i will play a song based on the Spotify features. 
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

SONG_FEATURES_PATH = r"/Users/James/Documents/Python/MachineLearningProjects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/pickles/my_features.pkl"


def add_play_count(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates a new play count column within an existing dataframe with "minutesTotal" and "duration" columns

    Args:
        dataframe (pandas dataframe): dataframe object to add play count column to

    Returns:
        pandas dataframe: original dataframe object with additional play count column
    """
    dataframe["playCount"] = dataframe["minutesTotal"] / dataframe["duration"]
    return dataframe


def plot_loss(model_history, exp: bool = False, save: bool = False):
    """creates a plot of the loss and validation loss for a given tensorflow fit history

    Args:
        model_history (type): tensorflow huistory object
        exp (bool, optional): plots an exponential y-axis. Defaults to False.
        save (bool, optional): save the file as log_loss.png in current directory. Defaults to False.
    """
    if exp:
        plt.plot(pd.DataFrame(np.exp(model_history.history["loss"])), label="loss")
        plt.plot(
            pd.DataFrame(np.exp(model_history.history["val_loss"])), label="val_loss"
        )
    else:
        plt.plot(pd.DataFrame(model_history.history["loss"]), label="loss")
        plt.plot(pd.DataFrame(model_history.history["val_loss"]), label="val_loss")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.legend()
    if save:
        plt.gca()
        save_dir = os.getcwd()
        name = "loss_plot.png"
        plt.savefig(os.path.join(save_dir, name), facecolor="w")
    plt.show()


def plot_hist(dataframe: pd.DataFrame) -> plt.Axes:
    """creates histogram plot for song playcount given an input dataframe

    Args:
        dataframe (pd.DataFrame): song features dataframe

    Returns:
        plt.Axes: histogram plot of playcount (seaborn style)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax = sns.histplot(data=dataframe, x="playCount", stat="count", kde=True)
    return ax


def save_model(model_to_save: keras.models.Sequential):
    """saves the specified model in the current directory as "trained_model.h5"

    Returns:
        saved model: keras model save object
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = "trained_model.h5"
    return model_to_save.save(os.path.join(current_dir, model_name))


def clean_songs_df(
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


#################################################### STILL TO REFACTOR ####################################################

X = streams_features.drop(
    ["artistName", "trackName", "minutesTotal", "trackId", "playCount"], axis=1
)
y = streams_features["playCount"]


standard_scalar = StandardScaler()
X = standard_scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=True
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.33, random_state=42, shuffle=True
)

model_params = {
    "optimizer": keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-06,
        amsgrad=False,
        name="Adam",
    ),
    "loss": tf.keras.losses.MeanAbsoluteError(),
    "hidden_activation": "relu",
    "output_activation": "relu",
    # "loss":keras.losses.Huber(),
    "initializer": tf.keras.initializers.HeNormal(),
    "regulizer": tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
}

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.05, patience=2, min_lr=0.00001
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, min_delta=0.001, restore_best_weights=True
)

model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=X_train.shape[1:], name="Input_Layer"),
        keras.layers.Dense(
            30,
            activation=model_params["hidden_activation"],
            kernel_initializer=model_params["initializer"],
            kernel_regularizer=model_params["regulizer"],
            name="Hidden_Layer1",
        ),
        keras.layers.Dense(
            10,
            activation=model_params["hidden_activation"],
            kernel_initializer=model_params["initializer"],
            kernel_regularizer=model_params["regulizer"],
            name="Hidden_Layer2",
        ),
        keras.layers.Dense(
            1, activation=model_params["output_activation"], name="Output_Layer"
        ),
    ]
)

model.compile(loss=model_params["loss"], optimizer=model_params["optimizer"])

history = model.fit(
    X_train,
    y_train,
    epochs=250,  # early stopping will kick in before 250 epochs
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=[reduce_lr, early_stopping],
    batch_size=16,
    shuffle=True,
)

plot_loss(history, exp=False)
model.evaluate(x=X_test, y=y_test, verbose=1, batch_size=16)

save_model(model)


# model = keras.models.load_model(model_file)

print(model.evaluate(X_test, y_test, batch_size=16))


if __name__ == "__main__":
    song_features_df = pd.read_pickle(SONG_FEATURES_PATH)
    song_features_df = add_play_count(song_features_df)
    columns_to_remove = [
        "type",
        "id",
        "uri",
        "track_href",
        "analysis_url",
        "time_signature",
    ]  # these features are outputs from how spotify catalogues the track, they won't help with prediction
    artists_tracks_to_drop = {
        "John Mayer": "On The Way Home"
    }  # this song has an incorrect duration and is returning a 30+ play count

    song_features_df = clean_songs_df(
        input_dataframe=song_features_df,
        columns_to_drop=columns_to_remove,
        artists_tracks_to_remove=artists_tracks_to_drop,
    )

