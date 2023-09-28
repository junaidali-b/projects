#Installing some basic libraries here and loading them. Only some of them might
#be needed.

#Installing & loading libraries for Hierarchical Modelling
install.packages("lme4")
install.packages("lmerTest")
install.packages("sjPlot")
install.packages("sjmisc")
install.packages("sjstats")
install.packages("arm")
library(lme4)
library(lmerTest)
library(sjPlot)
library(sjmisc)
library(sjstats)
library(arm)

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

#Installing tinytex for Quarto. Tinytex helps in outputting r markdown files
#in pdf format.Tinytex was giving multiple issues if I tried to download it
#through 'install.packages()' function. To solve this, the library had to be
#manually downloaded as a zip file and read into R.
tinytex:::install_prebuilt("C:/Users/Junaid Barodawala/Downloads/TinyTeX-2.zip")
library(tinytex)

#This helps me check if tinytex has been installed, and returns the file
#location of tinytex installed in the system.
tinytex_root()


#Installing formatting library for code chunks in Quarto.
install.packages("formatR")

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

save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/
           API Data/hiphop_main.rds")


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
save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/
           API Data/hiphop_main.rds")
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
load(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/
     API Data/hiphop_main.rds")

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


#Exploratory Analysis
summary(songdf)

#Using ggpairs to check for correlations. Since the number of variables in
#the data frame is very huge, it is difficult to understand anything from
#ggpairs. Hence a correlation matrix shall be used next.
ggpairs(dplyr::select(songdf, -c('duration', 'track', 'release', 'album')))


#Building a correlation matrix to check for all correlations

#These libraries are needed for creating the matrix
library(purrr)
library(Hmisc)

songdf_cor = Hmisc::rcorr(as.matrix(dplyr::select(
  songdf, -c('track', 'release',
             'releaseyr',
             'album')
)))

#The matrix returns other values as well, apart from just the correlation
#coefficients, and hence, only the correlation coefficients need to extracted
#and stored separately.

#Rounding off the correlation matrix to 3 decimals and storing it in a dataset
songdf_cor_r = data.frame(songdf_cor$r) %>% round(2)
str(songdf_cor_r)
songdf_cor_r$popularity

#Plotting correlation matrix using ggcorrplot. It is difficult to judge any
#kind of correlation from the correlation matrix plot, since there are very
#weak correlations in the data set. Hence this cannot be used in the report.
library(ggcorrplot)
ggcorrplot(
  songdf_cor_r,
  hc.order = TRUE,
  type = "lower",
  lab = TRUE,
  outline.col = "black",
  ggtheme = ggplot2::theme_gray,
  colors = c("black", "white", "#013220")
)


#Visually exploring correlations

#Age and Popularity
ggplot(data = songdf, aes(x = age, y = popularity)) +
  geom_point() +
  stat_smooth(method = "lm", col = "#1DB954") +
  xlab("Age of Songs (In Years)") +
  ylab("Song Popularity") +
  ggtitle("Relationship Between Ages of Songs and Popularity") +
  theme(plot.title = element_text(hjust = 0.5))
#scale_color_brewer(palette="Greens")

#Explicit and Popularity
ggplot(data = songdf, aes(x = explicit, y = popularity)) +
  geom_boxplot(fill = "#1DB954",
               colour = "black",
               width = 0.6) +
  stat_summary(
    geom = "text",
    fun = quantile,
    aes(label = sprintf("%1.1f", ..y..)),
    position = position_nudge(x = 0.40),
    size = 3.5
  ) +
  xlab("Explicit Lyrics") +
  ylab("Song Popularity") +
  ggtitle("Relationship Between Explicit Lyrics and Popularity") +
  theme(plot.title = element_text(hjust = 0.5))

#Speechiness and Popularity
ggplot(data = songdf, aes(x = speechiness, y = popularity)) +
  geom_point() +
  stat_smooth(method = "lm", col = "red") +
  xlab("Speechiness") +
  ylab("Song Popularity") +
  ggtitle("Relationship Between Speechiness and Popularity") +
  theme(plot.title = element_text(hjust = 0.5))


#Saving environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/
           Spotify Database/API Data/hiphop_main.rds")


#Basic Linear Modelling & Hypotheses Testing

#These basic linear models were built based on domain knowledge, however, they
#were not of much use.

#Linear model using Age, Explicitness and Speechiness, ignoring hierarchy
songdflm = lm(popularity ~ age + explicit + speechiness, data = songdf)
summary(songdflm) #Alpha=0.05
plot(songdflm)

#Linear model using only Age, removing other predictors
songdflm_null = lm(popularity ~ age, data = songdf)
summary(songdflm_null) #Alpha=0.05

anova(songdflm, songdflm_null) #Testing the difference between. Alpha=0.05

#Fixed Effect Model where each album is added as an interaction)
#Does not work well
songdffelm = lm(
  popularity ~ age + explicit + speechiness + age * album + explicit * album +
    speechiness * album,
  data = songdf
)
summary(songdffelm) #Alpha=0.05
#F-statistic: 49.39 on 1938 and 21283 DF,  p-value: < 2.2e-16

save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/
           API Data/hiphop_main.rds")

#Beginning Hierarchical Modelling- Increasing complexity bit by bit, and
#comparing models at every stage using ANOVA


#Simpler Models- Just Checking

#Random intercepts multilevel model- Explicit
songdflmer1 = lmer(popularity ~ explicit + (1 | album), data = songdf)
summary(songdflmer1)
coef(summary(songdflmer1))

#Random intercepts multilevel model- Explicit & Age
songdflmer2 = lmer(popularity ~ explicit + age + 
                     (1 |album), data = songdf)
summary(songdflmer2)
coef(summary(songdflmer2))

#Random intercepts multilevel model- Explicit, Age and Speeciness
songdflmer3 = lmer(popularity ~ explicit + age + speechiness + 
                     (1 | album), data = songdf)
summary(songdflmer3)
coef(summary(songdflmer3))
plot(
  songdflmer3,
  col = "#1DB954",
  main = "SongDF-LMER3",
  xlab = "Fitted Values",
  ylab = "Residuals"
)

#Comparing mixed effects model 2 and 3
anova(songdflmer2, songdflmer3)

#Comparing mixed effects model 1, 2 and 3
anova(songdflmer1, songdflmer2, songdflmer3)

#The models that were tried above were created using trial and error and did
#not work well enough. Since there are so many variables to choose from, the
#most appropriate ones are to be determined, by considering the most complex
#model first that has all the variables. Next, using the p values, the most
#insignificant fixed effect would be removed one at a time. Finally, the ANOVA
#test would be used to compare the models.

lmer0 = lmer(data = songdf, popularity ~ 1 + (1 |
                                                album)) #Baseline model
summary(lmer0)

lmer1 = lmer(
  data = songdf,
  popularity ~ explicit + age + speechiness + tempo + valence +
    danceability + energy + loudness + acousticness + instrumentalness +
    liveness + duration + (1 | album)
)

#This function helps in only extracting the fixed effects and intercept part of
#the summary, for mixed effect linear models
coef(summary(lmer1))

#Instrumentalness has been removed from the previous model
lmer2 = lmer(
  data = songdf,
  popularity ~ explicit + age + speechiness + tempo + valence +
    danceability + energy + loudness + acousticness + liveness +
    duration +
    (1 | album)
)
coef(summary(lmer2))

#Tempo has been removed from the previous model
lmer3 = lmer(
  data = songdf,
  popularity ~ explicit + age + speechiness + valence +
    danceability + energy + loudness + acousticness + liveness +
    duration +
    (1 | album)
)
coef(summary(lmer3))

#Song duration has been removed from the previous model
lmer4 = lmer(
  data = songdf,
  popularity ~ explicit + age + speechiness + valence +
    danceability + energy + loudness + acousticness + liveness +
    (1 | album)
)
coef(summary(lmer4))

#Loudness has been removed from the previous model
lmer5 = lmer(
  data = songdf,
  popularity ~ explicit + age + speechiness + valence +
    danceability + energy + acousticness + liveness + (1 |
                                                         album)
)
coef(summary(lmer5))

#Same as lmer2, excluding acousticness
lmer6 = lmer(
  data = songdf,
  popularity ~ explicit + age + speechiness + tempo + valence +
    danceability + energy + loudness + liveness + duration + (1 |
                                                                album)
)
coef(summary(lmer6))

#Comparing lmer 2, the model with statistically significant variables, with
#lmer6, the model wherein acousticness is removed
anova(lmer2, lmer6)

anova(lmer1, lmer2, lmer3, lmer4, lmer5, lmer6, test = 'F') %>% kable()

#This library helps in visualising and checking the performance of linear models
library(performance)

#Checking for normality in residuals
check_model(lmer2, check = c("linearity", "homogeneity"))

#Using sjPlot to visualise model lmer2
install.packages("sjPlot")
library(sjPlot)
sjPlot::plot_model(lmer2) #WORKED

#Inspecting random effects
install.packages('glmmTMB')
library(glmmTMB)
plot_model(lmer2, type = "re")

#Inspecting residuals
plot_model(lmer2, type = "resid")

#Inspecting slopes
plot_model(lmer2, type = "slopes")

icc(lmer2)

#QQ Plot- Checking if residuals follow a normal distribution
check_model(lmer2, check = c("qq", "normality"))

#Creating a table for lmer2. This table cannot be rendered using latex, and
#hence shall be discarded for the report
sjPlot::tab_model(
  lmer2,
  show.re.var = TRUE,
  pred.labels = c(
    "(Intercept)",
    "Explicit (True)",
    "Age",
    "Speechiness",
    "Tempo",
    "Valence",
    "Danceability",
    "Energy",
    "Loudness",
    "Acousticness",
    "Liveness",
    "Duration"
  ),
  dv.labels = "Effects of Song Features on Song Popularity"
)

#Making a table that would help in assessing model fit and goodness
model_performance(lmer2) %>% kable(caption = "Table to check performance and
                                   fit of the mixed effect linear model lmer2.")

save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")
