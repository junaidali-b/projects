#Libraries for accessing Spotify API in R. Devtools aids in installing
#libraries from GitHub,whenever required.
install.packages("devtools")
install.packages("spotifyr")
library(devtools)
library(spotifyr)

#Installing a version of spotifyr that is inspired by tidyverse. It will help
#us in interacting better with the API
install.packages('tinyspotifyr')
library(tinyspotifyr)

#Authentication for accessing Spotify API. The client ID and client secret are
#unique to each Spotify Application created on https://developer.spotify.com/

Sys.setenv(SPOTIFY_CLIENT_ID = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
Sys.setenv(SPOTIFY_CLIENT_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

access_token <- get_spotify_access_token()

#Testing API. The function 'get_artist_audio_features' returns the variables
#concerned with a particular musical artist. In this case, Eminem has been
#considered.
get_artist_audio_features('eminem') #succeeded


#Fetching details for 50 artists from hip-hop genre and storing it in a data
#frame. This dataframe will serve as the input dataframe for a for loop, that
#will download a list of all songs from Spotify, for each artist in the dataset
#rappers
rappers = get_genre_artists(
  genre = 'hip-hop',
  market = NULL,
  limit = 50,
  offset = 0,
  authorization = access_token
)

#Creating a dataframe with the desired format to serve as the main raw dataset.
hiphop = get_artist_audio_features('eminem')

#USE WITH EXTREME CAUTION! The following code will create a blank data frame
#and clean every row from the it, except for the headings.
hiphop = hiphop[0, ] #Clears up the whole database!

#Fetching details of all songs by all 50 artists in the list. A sleep time in
#seconds has been added to prevent rate limiter issues (Spotify Error 429).
#In case one does not add Sys.sleep (in other words, a cooldown time for the
#loop to run again, the loop creates and sends too many requests to the API,
#which causes the loop to stop working). There is an unknown number of requests
#that can be sent every 30 seconds, and every for loop dealing with the API has
# a random sleep time (whatever worked at the time!)
for (i in 1:50) {
  single_rapper_data = get_artist_audio_features(rappers$name[i])
  hiphop = rbind(hiphop, single_rapper_data)
  Sys.sleep(15)
}

#Despite adding sleep time, the API threw an error and out of 50 artists, it
#only executed the loop for 40.
tail(hiphop, 1) #Checking the last row of the incomplete dataset

#Checking row number of last successful loop execution, using the rapper's name,
#observed and obtained from the previous command.
which(rappers$name == 'The Kid LAROI')

#Attempting to fetch data of the remaining 10 artists. Sleep time has been
#increased to avoid rate limiter.
for (i in 41:50) {
  single_rapper_data = get_artist_audio_features(rappers$name[i])
  hiphop = rbind(hiphop, single_rapper_data)
  Sys.sleep(31)
}

#Checking if data has been fully fetched. Total number of unique artists should
#be 50

n_distinct(hiphop$artist_name) == n_distinct(rappers$name) #successfully matched

#Saving environment. In order to prevent loss of progress, the project
#environment has been saved to an rds file, that can be recovered whenever
#needed to continue where I left off.

save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")


#Right here,there was a problem, that the songs downloaded in the dataset
#'hiphop', was not having the dependent variable that was needed for the study,
#that is, 'popularity'. Upon checking the documentation, it was discovered that
#calling songs from a playlist provided the variable. Hence, the next step is
#to create playlists on Spotify, uploading songs from the dataset hiphop to
#them. Next, the features and details of the songs in these custom-made
#playlists would be extracted in a separate dataframe.

#Checking if user authorisation succeeded. In order to add songs to a playlist
#or create a playlist, we need to login as a user

get_my_playlists(
  limit = 20,
  offset = 0,
  authorization = get_spotify_authorization_code(),
  include_meta_info = FALSE
)

#Dividing Track URI column from Hip-Hop dataset, into chunks of 100, since the
#API only accepts URIs upto batches of 100. Each of these chunks will be passed
#into a for loop, that will upload songs to the specified Spotify playlist.
hiphop_track_uri_chunks = split(hiphop$track_uri,
                                ceiling(seq_along(hiphop$track_uri) / 100))

#The number 222 was arrived at, by dividing the total number of rows in the
#hiphop data frame, in chunks of 100. 222 were the number of chunks formed.
for (i in 1:222)
{
  add_tracks_to_playlist(
    '3rOI31k9heJ2qAM3WdnGSD',
    uri = unlist(hiphop_track_uri_chunks[i]),
    position = NULL,
    authorization = get_spotify_authorization_code()
  )
  Sys.sleep(15) #To prevent rate limiter issues (Spotify Error 429)
}

#The first loop only got executed till the 110th row of the list, since Spotify
#only allows upto 110*100=11000 songs in a playlist.Feeding remaining songs
#into second playlist. The following is a code to create a playlist in Spotify

create_playlist(
  '31akkdhbmbzen7qrnossiopmxzya',
  'Hip-Hop Songs 2',
  public = FALSE,
  collaborative = FALSE,
  description = 'All songs of 50 rappers. Songs: 11001 to 22000',
  authorization = get_spotify_authorization_code()
)


#Running loop again with interval 111:222. A third playlist might also be
#needed.

for (i in 111:222)
{
  add_tracks_to_playlist(
    '0hFOj9sU3J2WaEdHx160TM',
    uri = unlist(hiphop_track_uri_chunks[i]),
    position = NULL,
    authorization = get_spotify_authorization_code()
  )
  Sys.sleep(20) #To prevent rate limiter issues (Spotify Error 429)
}

#Creating 3rd playlist for remaining songs and adding the remaining songs to
#them. New interval is 221:222

create_playlist(
  '31akkdhbmbzen7qrnossiopmxzya',
  'Hip-Hop Songs 3',
  public = FALSE,
  collaborative = FALSE,
  description = 'All songs of 50 rappers. Songs: 22001 to 22186',
  authorization = get_spotify_authorization_code()
)


for (i in 221:222)
{
  add_tracks_to_playlist(
    '6wpEkFXqQKxti8HFSoGCyr',
    #got new playlist id from running previous code
    uri = unlist(hiphop_track_uri_chunks[i]),
    position = NULL,
    authorization = get_spotify_authorization_code()
  )
  Sys.sleep(20) #To prevent rate limiter issues (Spotify Error 429)
}

#Saving the environment (the next code snippet is risky)
save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

#Push code to git as well

#Fetching track details for each of the 3 playlists and storing them in a single
#dataframe, named 'hiphop_songs'. IDs for the 3 playlists can be obtained from
#Spotify.

hiphop_songs = get_playlist_audio_features(
  '31akkdhbmbzen7qrnossiopmxzya',
  #found in Spotify user profile
  c(
    '3rOI31k9heJ2qAM3WdnGSD',
    '0hFOj9sU3J2WaEdHx160TM',
    '6wpEkFXqQKxti8HFSoGCyr'
  ),
  authorization = get_spotify_access_token()
)

#The final raw dataframe with Track Popularity included,has been created.
#We'll be cleaning the data now. We'll be leaving the original data frame
#intact, and use a separate data frame to clean and work with.

songdf = hiphop_songs

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

#Continue to script 2 for data wrangling