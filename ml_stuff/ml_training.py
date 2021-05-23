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

streams_features_file = r"/Users/James/Documents/Python/Machine Learning Projects/Spotify_Listening_Analysis/Spotify 2.0/preprocessing/pickles/my_features.pkl"
streams_features_raw = pd.read_pickle(streams_features_file)
streams_features = streams_features_raw.copy()


def convert_duration(dataframe=streams_features):
    dataframe["duration"] = dataframe["duration_ms"].divide(60000)
    dataframe.drop("duration_ms", axis=1, inplace=True)
    return dataframe


def add_play_count(dataframe=streams_features):
    dataframe["playCount"] = dataframe["minutesTotal"] / dataframe["duration"]
    return dataframe


def plot_loss(model_history, exp=False):
    if exp:
        plt.plot(pd.DataFrame(np.exp(model_history.history["loss"])), label="loss")
        plt.plot(
            pd.DataFrame(np.exp(model_history.history["val_loss"])), label="val_loss"
        )
    else:
        plt.plot(pd.DataFrame(model_history.history["loss"]), label="loss")
        plt.plot(pd.DataFrame(model_history.history["val_loss"]), label="val_loss")
    fig = plt.gcf()
    plt.grid(True)
    # plt.gca().set_ylim(0, 10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.legend()
    plt.show()
    my_path = os.path.dirname(os.path.abspath(__file__))
    my_file = "loss_plot.png"
    fig.savefig(os.path.join(my_path, my_file))


def plot_hist(model):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax = sns.histplot(data=model, x="playCount", stat="count", kde=True)
    return ax


def save_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = "trained_model.h5"
    model.save(os.path.join(current_dir, model_name))


columns_to_drop = ["type", "id", "uri", "track_href", "analysis_url", "time_signature"]
# dropping unhelpful features
streams_features.drop(columns_to_drop, inplace=True, axis=1)
streams_features = convert_duration(streams_features)
streams_features = add_play_count(streams_features)
streams_features = streams_features[
    (streams_features["artistName"] != "John Mayer")
    & (streams_features["trackName"] != "On The Way Home")
]  # this one song has an incorrect duration and is returning a 30+ play count, definitely something not right

# Cap songs plays at 10
streams_features["playCount"].where(
    streams_features["playCount"] <= 10, 10, inplace=True
)

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
        keras.layers.Dense(
            30,
            activation=model_params["hidden_activation"],
            input_shape=X_train.shape[1:],
            kernel_initializer=model_params["initializer"],
            kernel_regularizer=model_params["regulizer"],
        ),
        keras.layers.Dense(
            10,
            activation=model_params["hidden_activation"],
            kernel_initializer=model_params["initializer"],
            kernel_regularizer=model_params["regulizer"],
        ),
        keras.layers.Dense(1, activation=model_params["output_activation"]),
    ]
)

model.compile(loss=model_params["loss"], optimizer=model_params["optimizer"])

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

save_model()