# Hierarchical Linear Modelling on Spotify Audio Features for Hip-Hop Category

Please read this file to understand how the scripts have been confirgured and in which order the files need to be executed.

The project is divided into 4 scripts and are to be executed in the following order:
1. **Dataset Creation From Spotify API:**
	- The data for this project is not available directly through a CSV and it had to be fetched directly using Spotify's API integration with R.
	- It is highly recommended to run snippets of code in parts, instead of running the entire script. This is because the Spotify API may blacklist your access if too many requests are bombarded at the same time on in quick succession. Some system sleep timers have been put in place to prevent this however a slow and steady pace is recommended.
	- A Spotify developer account and relevant API keys are required for running this script. This is available at- <https://developer.spotify.com/documentation/web-api>
	- The resulting environment after completion of this script has been saved in an RDS file, which would be required to run subsequent scripts. This file is also available in this repository tree and is named "hiphop_main.rds".
	
2. **Data Wrangling:**
	- This script wrangles and cleans the data, and starts with loading the RDS file created from the first script.
	- It is to be noted that if the "hiphop_main.rds" file is directly downloaded from the git repository, you may skip this step as it already contains clean wrangled data.
	- Similar to the first script, here it is recommended to run the script in parts rather than executing the entire file together.
	- Cleaned data and variables have been updated to the RDS file, saving the environment for the next script.
3. **Exploratory Analysis:**
	- The RDS file from the previous step (or the downloaded one), has been loaded at the beginnning of this sript.
	- This script contains visualisations of the cleaned data to help in understanding the data and possible linear correlations that it may contain.
	- This step is not mandatory but it highly recommended before beginning the modelling process.
4. **Linear and Hierarchical Linear Modelling:**
	- This is the last step of the project, and this script, similar to others, must be executed after the 'hiphop_main.rds' file has been loaded and has updated the environment.
	- The models have been created using simple linear modelling first, and hierarchical linear models have been created next.
	- Lastly, all models created have been compared to each other, and is it suitable to use visualisations and tables from this script and the 3rd script for creating reports, when necessary.

**Note:** Your R installation may or may not have the packages required for running the scripts. In case the packages do not exist they need to be installed using this syntax:
install.packages('package')

Software required for this project:
1. R (base)
2. R Studio
3. Git (Optional for version control)
