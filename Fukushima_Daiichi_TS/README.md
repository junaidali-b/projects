#Instructions for Fukushima Daiichi Time Series Modelling Project

Please read this file to understand how the scripts have been confirgured and in which order the files need to be executed.

The project is divided into 3 scripts and are to be executed in the following order:
1. **Data Wrangling & Processing:** 
	- This script opens up raw data and completes the first step, that is preprocessing and cleaning the data. 
	- The raw data is accessed from CSV files accessed from ...
2. **Harmonic Computation and Time Series Modelling:** 
	- The environment prepared from running the first script has been stored in a RDS file, and the same has been loaded at the beginning of this file.
	- This script is responsible for all trigonometric computations and time series models using ARIMA (Autoregressive Integrated Moving Average) technique.
	- The RDS file has been updated at the end of this script in order to run the next script. The updated environment contains the required models for running the last script.
3. **Model Comparison and Plots:**
	- This is the last script in the project, and is followed by the script for modelling. At the beginning of this script, the RDS file responsible for preserving the environment needs to be loaded.
	- In this script, the models built in the second script have been compared and the results have been visualised. This script would be helpful for building reports when necessary.
	
**Note:** Your R installation may or may not have the packages required for running the scripts. In case the packages do not exist they need to be installed using this syntax:
install.packages('package')

Software required for this project:
1. R (base)
2. R Studio
3. Git (Optional for version control)
