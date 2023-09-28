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

#Reading data for chrolophyll
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

#Chunk for performing all operations apart from plots (guts of the code)
#This chunk must run before model related plots start

#Core computations

mod1.sel <- c(NA,0,NA,NA,NA,NA,NA,NA,NA,NA,NA,0,0,NA)

boxmean.ts <- chl_ts_m
month.start <- start(boxmean.ts)[2]
idx.n <- seq(1,length(boxmean.ts))
idx.n.use <- idx.n[boxmean.ts <= 0]  #  identify the zeros
idx.tn.use <- idx.n[boxmean.ts > 0] 

# try replacement of zeros with a relevant small number...
boxmean.ts[idx.n.use] <- 10^mean(log10(boxmean.ts[idx.tn.use]))   

# replace with min of actual recorded data
boxmean.ts <- log10(boxmean.ts)
#
boxmean2.ts <- boxmean.ts
boxmean2.ts[idx.n.use] <- NA   # replace with min of actual recorded data

# Set up Dominant Frequency State Analysis parameters
# to model inputs of
#   a) 6 month cycle 
#   b) 1 year cycle
#   c) a linear time trend 
xtu <- boxmean.ts

order.arima <- c(1,0,0)          #  AR(1)
order.sarima <- c(1,0,0)         #  SAR(1)
t.seq <- seq(0,1-(1/length(xtu)),1/length(xtu))  # linear trend component
# Harmonic terms
# ts.tm <- seq(1,length(xtu))
ts.tm <- seq(month.start,length(xtu)+(month.start-1))
f1 <- 1/(6)    #6 months
h.c1 <- cos(2*pi*f1*ts.tm)
h.s1 <- sin(2*pi*f1*ts.tm)
f2 <- 1/(12)    #12 months
h.c2 <- cos(2*pi*f2*ts.tm)
h.s2 <- sin(2*pi*f2*ts.tm)
f3 <- 1/(12*10)    #10 year
h.c3 <- cos(2*pi*f3*ts.tm)
h.s3 <- sin(2*pi*f3*ts.tm)
f4 <- 1/(4)    #4 month
h.c4 <- cos(2*pi*f4*ts.tm)
h.s4 <- sin(2*pi*f4*ts.tm)
f5 <- 1/(3)    #3 month
h.c5 <- cos(2*pi*f5*ts.tm)
h.s5 <- sin(2*pi*f5*ts.tm)

# Set up data /  parameter settings
drivers.df <- data.frame(h.c1,h.s1,h.c2,h.s2,h.c3,h.s3,h.c4,h.s4,h.c5,h.s5,t.seq)  # add / remove relevant inputs
drivers.order <- list(c(0,0),c(0,0),c(0,0),c(0,0),c(0,0),c(0,0),c(0,0),c(0,0),c(0,0),c(0,0),c(0,0)) # (a,b) a= AR order, b= MA order (-1 none)

# Set the parameter fields to be estimated
fixedB.pars <- mod1.sel

#Estimate the frequency states
arimax1 <- 
  arimax(xtu,order=order.arima,seasonal=list(order=order.sarima,period=12),
         xtransf=drivers.df,transfer=drivers.order,fixed =fixedB.pars,method='ML')

arimax1.fit <- fitted(arimax1)     # fitted values
arimax1.res <- residuals(arimax1)  #  residual values

rsquared <- 1 - (arimax1$sigma)/var(xtu)   #r-squared value

# print out 95% Confidence Intervals of parameters
coefs2s <- cbind(coefficients(arimax1),confint(arimax1,level=0.95))  #plot coef & 95% CI

##############################################################################
#Extract the Annual model harmonic cycle
tsC.tm <- seq(1/52,12,length.out=52)
f1 <- 1/(6)    #6 months
hc.c1 <- cos(2*pi*f1*tsC.tm)
hc.s1 <- sin(2*pi*f1*tsC.tm)
f2 <- 1/(12)    #12 months
hc.c2 <- cos(2*pi*f2*tsC.tm)
hc.s2 <- sin(2*pi*f2*tsC.tm)
f4 <- 1/(4)    #4 month
hc.c4 <- cos(2*pi*f4*tsC.tm)
hc.s4 <- sin(2*pi*f4*tsC.tm)
f5 <- 1/(3)    #3 month
hc.c5 <- cos(2*pi*f5*tsC.tm)
hc.s5 <- sin(2*pi*f5*tsC.tm)

mean.level <- mean(boxmean.ts)
tot.climatology <- coefs2s[4,1]*hc.c1 + coefs2s[5,1]*hc.s1 + coefs2s[6,1]*hc.c2 + coefs2s[7,1]*hc.s2 + coefs2s[10,1]*hc.c4 + coefs2s[11,1]*hc.s4 + coefs2s[12,1]*hc.c5 + coefs2s[13,1]*hc.s5 + mean.level

caesium137 = plankton_cs137$bq

# fit the ARIMA model with xreg
arima2 <- auto.arima(chl_ts, xreg = bq_ts, seasonal = T)
arimax2 = arima2

arima2_fit = fitted(arima2)     # fitted values
arima2_res <- residuals(arima2)  #  residual values

#R squared
rsquared2 <- 1 - (arima2$sigma)/var(chl_ts)

# print out 95% Confidence Intervals of parameters
arima2_coef <- cbind(coefficients(arima2),confint(arima2,level=0.95)) 

#Plot raw data and incident
ggplot(data = chl, aes(x = date, y = chlorophyll)) + 
  geom_line() + 
  labs(title = "Time Series Plot", 
       x = "Years (From 1998 to 2021)",
       y = "Observed Value of Chlorophyll")+
  geom_vline(xintercept = as.Date("2011-03-11"), 
             color = "red", 
             linetype = "dashed")+
  theme(plot.title = element_text(hjust = 0.5, 
                                  face = "bold"))

aggregate(Value ~ iso, data = rad, FUN = sum) %>% 
  kable(caption = "Summary of Radioactive Isotopes",
        col.names = c("Isotope","Value in Becquerels (Bq)"))

#Plotting raw data of Cs137

ggplot(data = plankton_cs137, aes(x = date, y = bq)) + 
  geom_line() + 
  labs(title = "Measured Radioactivity from Caesium-137 in Becquerels (Bq)", 
       x = "Years (From 2011 to 2022)",
       y = "Becquerels (Bq)")+
  geom_vline(xintercept = as.Date("2011-03-11"), 
             color = "red", 
             linetype = "dashed")+
  theme(plot.title = element_text(hjust = 0.5, 
                                  face = "bold"))

tsdiag(arimax1)
mtext(expression(bold("Diagnostic Plots for arimax1")), 
      side = 3, 
      line = -1.5, 
      outer = TRUE)

tsdiag(arimax2)

cbind(AIC(arimax1, arimax2),
      rsquared = c(rsquared,rsquared2)) %>% 
  kable(cap = "Comparison of Time Series Models")


arimax2_coef = as.data.frame(tidy(coeftest(arimax2)))
arimax2_coef[,c(4,5)] = round(arimax2_coef[,c(4,5)],3)
arimax2_coef %>% kable(cap = "Summary of Coefficients")


# plot fitted model 
ts.plot((chl_ts), arima2_fit,
        gpars=list(xlab="Months from January to December 2021",
                   ylab="Observed Value of Chlorophyll",
                   main="Fitted Transfer Function Model (2011)",
                   log='y',lwd=c(1,2.3),xlim=c(2011,2012),
                   cex=1.1,cex.main=1.2,lty=c(1,1),col=c(1,4)))
for (i in 2011:2012)
{abline(v=i,lty=2,col="grey")}
abline(v="2011.3", lty=2, col="red")
abline(h=0)

# plot fitted model 
ts.plot(as.numeric(chl_ts), 
        arima2_fit,
        gpars=list(xlab="Years from 2011 to 2021",
                   ylab="Observed Value of Chlorophyll",
                   main="Fitted Transfer Function Model (2011)",
                   log='y',
                   lwd=c(1,2.3),
                   cex=1.1,
                   cex.main=1.2,
                   lty=c(1,1),
                   col=c(1,4)))
for (i in 2011:2021)
{abline(v=i,lty=2,col="grey")}
abline(v="2011.3", lty=2, col="red")
abline(h=0)