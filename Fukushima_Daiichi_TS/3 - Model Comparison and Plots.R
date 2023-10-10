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
#Continuation after Harmonic Comutation and Modelling
load(file = "D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima.rds")

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

#Saving the environment
save.image(file = "D:/R Studio/Downloads/Data/For Project/507/Fukushima/Fukushima.rds")