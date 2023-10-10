Instructions for Machine Learning project to Predict Queen Bee Acceptance and Presence.

#Please read through this document to understand the order in which the syntax is to be executed for replication.

#Storage & Working Directory

All scripts are to be stored in a subfolder called '/Scripts' within a main folder called 'Honey Bee'
All backups are to be stored in a subfolder called '/Backups' within a main folder called 'Honey Bee'
The Quarto file needs to be stored in a subfoler called '/Quarto' within the main folder called 'Honey 'Honey Bee'
The main folder 'Honey Bee' is to be set as the working directory for all python scripts
The subfolder 'Quarto' is to be set as the working directory for all QMD and R related files.
#Required Software:

Python 3.11.3- https://www.python.org/downloads/
R- https://cran.r-project.org/bin/windows/base/
Spyder IDE: https://www.spyder-ide.org
R Studio: https://posit.co/download/rstudio-desktop/
Anaconda: https://www.anaconda.com/download
Required Data:

Raw Data: https://www.kaggle.com/datasets/annajyang/beehive-sounds
Pickle Backup (Cleaned Data, Variables and Subsets): https://drive.google.com/file/d/10JJwmoGKbkbUr4fg0QgiFRmRsgPDP82h/view?usp=drive_link (Please note that it is important to load this Pickle file into Python and R, to execute any kind of code apart from the initial data wrangling. The Pickle file is to be loaded at at the beginning of all scripts to avoid any errors.)
#Construction of conda environment:

A conda environment is to be created by the name of 'spyder' and is to be set up in the working directory of the spyder IDE
The following packages are need to be installed for the scripts in the project to work
numpy
pandas
sklearn
PIL
gc
os
matplotlib
plotnine
xgboost
seaborn
cv2
umap
tensorflow (recommended along with all add-ons)
#Please note that some of these libraries may come preinstalled along with Python installation. #All of these libraries can be installed onto the environment 'spyder' using either Anaconda graphical interface or conda command line.

#Syntax to install libraries using conda command line

conda activate spyder
conda install -c conda-forge package
Alternatively syntaxes that may be entered after activating conda environment:

conda install package
pip install package
#Commands to start Spyder using specific enviroment (to be inputted within Anaconda command line)

conda activate spyder #This activates the environment named Spyder
spyder #This starts Spyder IDE (In the prompts above, do not input the comments added after the # symbol)
#The scripts in the Honey Bee project are to be executed in this order:

Data Wrangling.py
Basic Visualisation.py
UMAP.py
Support Vector Machine.py
Random Forest.py
Gradient Boosting.py
CNN.py ('Pickle Backup.py' only stores a function to simplify creating and updating pickle archive files. It need not be executed specially as it has been executed separately in other files.)
#The QMD file can be opened directly in R Studio.

#Libraries required for running QMD file:

reticlulate (EXTREMELY IMPORTANT TO RUN PYTHON SYNTAX IN R)
knitr
#Syntax to install libraries in R Studio

install.packages("package")
#In each of the scripts, file locations have been set based on how the files have been stored in my PC, however, there may be a case wise need to edit the file locations as needed. They may differ in every computer, and hence, it is important to check and edit the locations referenced in each script as required.