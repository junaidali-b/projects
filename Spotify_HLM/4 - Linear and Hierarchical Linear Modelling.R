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

#Retrieving environment from RDS file
#Continuation after exploratory analysis
load(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

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

save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")

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

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/Spotify Database/API Data/hiphop_main.rds")