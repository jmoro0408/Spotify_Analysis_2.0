# Spotify Analysis

**Version 2.0.0**

This is my second attempt at creating a project to analyze my Spotify listening history, and use the Spotify API to develop a machine learning algorithm that will predict how much I will listen to any inputted song. 

This README file is a short description of how the project was undertaken and the steps involved. 


## Data Gathering
### My listening history

The first step to analyzing any Spotify listening history is to request your data from Spotify. This can be done by logging into your Spotify account on the web and going to "Privacy Settings" -> "Download your Data". 

This will send a request to Spotify to send your recent listening history. This usually takes a few days to a week, and unfortunately only includes the last 12 months listening history. Luckily I listen to Spotify a lot so this was a lot of tracks (around 9000). 

### Rolling Stones top 500 Albums
This one is simple, I just used a premade csv file from Kaggle that includes the Rolling Stones Top 500 Albums, Artists, Year of release Genre, and Sub-Genre. 
[You can find the Kaggle dataset here][rolllingstonesdataset]. 


## Data Preprocessing
### Listening History
Now we have our listening history from Spotify we need to preprocess it and turn it into something useful for analysis. 

First thing, the listening history is supplied as several json files, bundled with a log of other information that isn't used through this project, but is pretty interesting and may be the basis of future projects. 

The info provided by Spotify is:
- Playlist - A summary of the playlists created or saved, and any songs saved
- Streaming History - A list of items (e.g. songs, videos, and podcasts) listened to or watched in the past year
- Your Library - A summary (at the point of the date of the request) of the content saved in Your Library (songs, podcasts, and videos)
- Search queries - A list of searches made
- Follow - Information regarding who you follow and who follows you
- Payments - This includes details of the payment method
- User Data - Personal User Data including name, address, email address etc
- Inferences - Insights about your interests for ad targeting 


More details regarding each json and what is is included can be found [here][spotofysupplieddata]. 

Obviously some of this information is personal and should not be shared, therefore I've added it all to my .gitignore file and haven't included any of the raw data on my GitHub. 

The preprocessing/my_data_preprocessing.py file includes all the code used to turn the data listening history data provided by Spotify into something useful for analysis. In short:

1. Combine all listening history jsons into a single dataframe
2. Remove unwanted tracks (podcasts, white noise tracks etc)
3. Remove tracks with "Unknown Song" and "Unknown Artist". How Spotify doesn't now what was streamed is unknown to me..
4. Combine songs with more than one play into a single row and sum their total play time. 
5. Save dataframe to a pickle file for later. 


### Rolling Stones 500
Not much preprocessing required here as the supplied csv is is a good state. 

The file for all the RollingStones500 PReprocessing is included in preprocessing/rollingstones500_preprocessing.py. In short:

1. Import csv to pandas dataframe
2. drop columns: ["Number", "Genre", "Sub-genre"]
3. Save dataframe to pickle

## Feature Retrieval

Now we have all the songs we need to do the analysis we need to use the Spotify API to get song features. 

Spotify provides the following features for every song in their library:

danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness , liveness, valence, tempo, type, id, uri, track_href, analysis_url, duration_ms, time_signature. 

Some of these are self-exploratory, information on each of the features can be found [here][spotifyfeaturesinfo].

The gathering of track ID and features for each track are relatively slow (around 15 songs/second) and so I wrapped a tqdm progress bar around each function to check on how they are performing. 

![tqdm progress bar][tqdmimage]


### Listening History


The code for my my listening history can be found at preprocessing/my_data_spotify_feature_retrieval.py. 

The file essentially uses the Spotipy library a a wrapper around the Spotify developers API. 


1. Uses the artist and track name to return the track ID from Spotify
2. Applies this function to all tracks in the dataframe (missing tracks are assigned nan)
3. Using the track ID to find all the features for each track and append them to the dataframe. 
4. Save the completed dataframe to a pickle for later. 

### Rolling Stones 500

The code for the feature retrieval of the rollingstones 500 dataframe can be found at: preprocessing/rollingstones500_feature_retrieval.py 


I used a lot of the same functions from the listening history feature retrieval, as is it essentially the same task: Get track ID's based on artist and track -> get features from track ID. 

The main difference is the RollingStones 500 CSV only provides the album name and artist name, so we first have to go and pull the track names for that album before getting track ID and features. 

## ML Training

Now we have the listening history and rolling stones 500 songs with their unique features we an start doing some machine learning prep. 

The plan is to use the listening history song features as our "X" variable, and the total play count for each song as the target "y" variable. 

The rolling stones dataframe will then be used to make predictions on. 

## Listening History







# Future Project Plans
- Request more data from Spotify (another 6 months worth) and combine to increase the size of my data set. 
- Tweak the algorithm so it is not as conservative and provides more outgoing results (gathering more data will likely help with this). 
- I would like to add the Rolling Stones Top 500 songs (perhaps with Spotify features) as a Kaggle Dataset for others to play around with, but first need to confirm which albums are missing, and also any issues/legal implications with publishing Spotify feature data. 



[rolllingstonesdataset]: https://www.kaggle.com/notgibs/500-greatest-albums-of-all-time-rolling-stone
[spotofysupplieddata]: https://support.spotify.com/us/article/understanding-my-data/
[spotifyfeaturesinfo]: https://developer.spotify.com/documentation/web-api/reference/#object-audiofeaturesobject
[tqdmimage]: https://raw.githubusercontent.com/jmoro0408/Spotify_Analysis_2.0/main/preprocessing/rollingstones_tqdm_example.jpg "TQDM Progress bar example"