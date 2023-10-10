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
library(gt)
library(skimr)
library(ggdist)
library(showtext)

#Retrieving environment from RDS file
#Continuation after data wrangling
load(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

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

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

#Continue to script 4 for Linear and Hierarchical Linear Modelling