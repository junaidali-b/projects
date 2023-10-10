#Libraries for Data Wrangling & Visualisation
library(tidyverse)
library(magrittr)
library(dplyr)
library(ggplot2)
library(knitr)
library(GGally)
library(httr)
library(jsonlite)
library(tidyr)
library(zoo)
library(RCurl)

#Other miscellaneous libraries
install.packages('gt')
install.packages('skimr')
install.packages('ggdist')
install.packages('showtext')
library(gt)
library(skimr)
library(ggdist)
library(showtext)

#Retrieving environment from RDS file
#Continuation after dataset creation from API
load(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

#Checking the structure of the new data frame
str(songdf)

songdf %>% summarise_all(funs(sum(is.na(.))))
#Checked for Missing Values. None Found

#Checking just the column names of songdf
colnames(songdf)

#Removing unwanted columns
songdf = select(
  songdf,
  -c(
    "playlist_id",
    "playlist_name",
    "playlist_img",
    "playlist_owner_name",
    "playlist_owner_id",
    "is_local",
    "primary_color",
    "added_by.href",
    "added_by.id",
    "added_by.type",
    "added_by.uri",
    "added_by.external_urls.spotify",
    "track.disc_number",
    "track.episode",
    "track.href",
    "track.is_local",
    "track.preview_url",
    "track.track",
    "track.track_number",
    "track.type",
    "track.album.album_type",
    "track.album.available_markets",
    "track.album.href",
    "track.album.id",
    "track.album.images",
    "track.album.release_date_precision",
    "track.album.total_tracks",
    "track.album.type",
    "track.album.external_urls.spotify",
    "track.external_ids.isrc",
    "track.external_urls.spotify",
    "video_thumbnail.url",
    "key_name",
    "mode_name",
    "key_mode",
    "track.id",
    "analysis_url",
    "time_signature",
    "added_at",
    "track.album.artists",
    "track.artists",
    "track.album.uri",
    "track.uri",
    "key",
    "mode",
    'track.available_markets'
  )
)

colnames(songdf) #Checking column names

#Retrieving environment from RDS file (cause I had to restart R)
load(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

#Renaming Columns

library(plyr) #Using dplyr isn't working, trying plyr instead

songdf = plyr::rename(
  songdf,
  c(
    'track.duration_ms' = 'duration',
    'track.explicit' = 'explicit',
    'track.name' = 'track',
    'track.popularity' = 'popularity',
    'track.album.name' = 'album',
    'track.album.release_date' = 'release'
  )
)

colnames(songdf) #Checking column names

#Manipulating column formats. This will help in working with the data later,
#for linear modelling

#Converting duration in milliseconds to seconds
songdf$duration = songdf$duration / 1000

#Extracting year from date column.
install.packages("stringr")
library(stringr)

#Split date string by -, extracted first element from nested list of each row,
#and converted sub-strings to integers. This is done in order to extract the
#release year, which will later be used to calculate the age of the songs.
songdf$releaseyr = as.numeric(lapply(str_split(songdf$release, "-"), '[[', 1))
songdf %>% summarise_all(funs(sum(is.na(.)))) #Checking for missing vales
head(select(songdf, c(duration, releaseyr)))

#Adding a column for age (how old is the song)

songdf$age = 2022 - songdf$releaseyr

#Checking if column has been correctly computed. This would return NA if there
#were missing values
mean(songdf$age)

#Reordering columns as per hierarchy and usefulness of columns.
#Hierarchy: Genre (Hip-Hop)-> Date/Year/Age -> Albums-> Tracks.

songdf = songdf[, c(
  "release",
  "releaseyr",
  "age",
  "album",
  "track",
  "popularity",
  "explicit",
  "speechiness",
  "tempo",
  "valence",
  "danceability",
  "energy",
  "loudness",
  "acousticness",
  "instrumentalness",
  "liveness",
  "duration"
)]

#Using this function to check the order of columns in the data frame songdf
colnames(songdf)

#Exporting data frame into a csv for sending data to quarto. This was done
#because the project was made in this r-script file, but the report is to be
#created in Quarto. Unfortunately, Quarto cannot inherit the global environment
#of the project, while rendering the document, and hence the data frame needs
#to be defined separately. The csv would be used for this definition in Quarto.
write.csv(songdf, "D:/R Studio/r-studio-main/songdf.csv", row.names = FALSE)

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

#Continue to script 3 for exploratory analysis