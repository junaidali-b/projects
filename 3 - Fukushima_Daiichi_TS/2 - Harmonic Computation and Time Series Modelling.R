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

#Retrieving environment from RDS file
#Continuation after data wrangling and preprocessing
load(file = "D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima.rds")

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

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima.rds")

#Continue to script 3 for Model Comparison and Plots