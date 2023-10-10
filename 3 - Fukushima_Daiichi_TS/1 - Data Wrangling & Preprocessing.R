#Loading required packages
library(TSA)
library(forecast)
library(ggplot2)
library(tidyverse)
library(knitr)
library(dplyr)
library(zoo)
library(lmtest)
library(broom)

#Reading data for chlorophyll
chl = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Plankton/dataCHLA_NASAcombo.csv")

#Reading data for radioactivity
rad11 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2011.csv")
rad12 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2012.csv")
rad13 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2013.csv")
rad14 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2014.csv")
rad15 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2015.csv")
rad16 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2016.csv")
rad17 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2017.csv")
rad18 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2018.csv")
rad19 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2019.csv")
rad20 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2020.csv")
rad21 = read.csv("D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima Radioactivity/Extracted/2021.csv")

#Data Wrangling

#renaming columns
colnames(chl)[1] = "date"
colnames(chl)[2] = "chlorophyll"

#chlorophyll date conversion
chl$date = as.Date(chl$date)

#Combining radiation data frames
rad = rbind(rad11, rad12, rad13, rad14, rad15, rad16, rad17, rad18, rad19, rad20, rad21)

#To remove missing values in the measured readings column
rad = subset(rad, Value != "ND")

rad$Value = round(as.numeric(rad$Value),2)

colnames(rad)[2]="iso"
colnames(rad)[9]="station"
colnames(rad)[10]="date"

rad$date = as.Date(rad$date)
rad$date = format(rad$date, "%Y-%m-%d")

#filter for getting values for cs137 from station T-1
cs137_filter = rad$iso=="Cs-137" & rad$station=="T-1"

#new dataframe for Cs-137
cs137 = data.frame(date = rad$date[cs137_filter],
                   bq = rad$Value[cs137_filter])
cs137$date = as.Date(cs137$date)

#summing up readings on the same date
cs137 = aggregate(bq ~ date, cs137, sum)
cs137$date = format(cs137$date, "%Y-%m-%d")

# Extract the chlorophyll values for the matching dates
cs137$chl = ifelse(cs137$date %in% chl$date, 
                   chl$chlorophyll[chl$date %in% cs137$date], 
                   NA)

#Creating date vector for all dates in rad
all_dates <- seq.Date(min(chl$date),
                      max(chl$date),
                      by = "day")

#Changing format of all dates
#all_dates = format(all_dates, "%Y-%m-%d")

#Interpolate values for chlorophyll
interpolated_chl <- approx(chl$date, chl$chlorophyll, all_dates)$y

#Creating new dataset
plankton = data.frame(date = all_dates, chl = interpolated_chl)

# Extract the radioactivity values for the matching dates
plankton$bq = ifelse(plankton$date %in% cs137$date,
                     cs137$bq[plankton$date %in% cs137$date],
                     0)

#plankton$date = format(plankton$date, "%Y-%m-%d")

# loop through each row in plankton and try to match it with a row in cs137
for (i in 1:nrow(plankton)) {
  
  index = which(plankton$date[i] == cs137$date)
  # if a match is found, add the value from cs137 to the matched_values vector
  if (length(index) > 0) {
    plankton$bq[i] <- cs137$bq[index]
  }
}

#checking if there are any NA values- there were none
#sum(cs137$bq)
#finally worked!

# Find the min and max date in cs137
cs137_min_date <- min(cs137$date)
cs137_max_date <- max(cs137$date)

# Create a subset of plankton dataset
plankton_cs137 <- subset(plankton, 
                         date >= cs137_min_date & date <= cs137_max_date)

#Rounding values of chlorophyll to 3 decimal places
plankton_cs137$chl = round(plankton_cs137$chl, 3)

plankton_cs137$bq[plankton_cs137$bq == 0] = NA

#Interpolate values for cs137
interpolated_bq <- round(approx(plankton_cs137$date, 
                                plankton_cs137$bq, 
                                plankton_cs137$date)$y,1)

plankton_cs137$bq = interpolated_bq

# Convert columns to a time series objects (for radioactive)
chl_ts = ts(plankton_cs137$chl, 
            frequency = 365, 
            start = c(2011,4,1), 
            end = c(2021,12,15))

bq_ts = ts(plankton_cs137$bq, 
           frequency = 365, 
           start = c(2011,4,1),
           end = c(2021,12,15))

#Log transforming radioactivity values
which0 = which(bq_ts<=0)
bq_ts[which0]=0.1 #replacing zeros
#bq_ts=log10(bq_ts) #log transformation

chl_ts_m = ts(chl$chlorophyll,
              frequency = 12,
              start = c(2011,04),
              end = c(2021,12))

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima.rds")

#Continue to script 2 for Harmonic Computations and Time Series Modelling