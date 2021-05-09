# Listening_History_Analysis.py
Small project to analyse some of my Spotify listening habits. 
Check out the notebook for some commentary around my rather embarrassing listening habits

The Spotify Listening_History_Analysis.py file outputs a .pkl file that contains a dataframe describing 8500 songs I've listened to broken down into:
1) The Artist
2) The Track Name
3) The Spotify Track ID
5) How many Minutes I've listened
6) The songs "Features" as described by Spotify. 

The Spotify Features are described by the Spotify Devloper API as follows:
 1) Danceability: Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.
 2) Valence: Describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
 3) Energy: Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.
 4) Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece, and derives directly from the average beat duration.
 5) Loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks.
 6) Speechiness: This detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.
 7) Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”.
 8) Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.
 9) Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
 10) Key: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.
 11) Mode: Indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
  
  
#RollingStone_Top500_Features.py
  
  This file takes a CSV of the Rolling Stones Top 500 Albums (dataset from Kaggle - https://www.kaggle.com/notgibs/500-greatest-albums-of-all-time-rolling-stone)
  and uses a similar method as the listening history analysis file to assign the same spotify features for each track in each album. 
  The dataframe is then saved as a .pkl file for later use.
  
  #Spotify_ML.py
  Nothing here yet. The plan is to create a machine learning algortihm that looks at the features of my listening history dataset, along with the minutes listened and then predict how much I will like songs from the rolling stones 500 list. Hopefully I'll find some new music!
  
  