#The .spydata archive created after data wrangling, has been loaded first.

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

#Cross-checking data and audio files segments and level of detail
audio_names_sample = os.listdir("D:/Python/Spyder/Honey Bee/Data/spec_files/1")

#Getting unique values from queen related columns
np.unique(hivedata["queen_presence"])
np.unique(hivedata["queen_status"])
np.unique(hivedata["queen_status"])

# Load an audio file
audio_path = hivedata.loc[0, 'audio_0']
audio, sr = librosa.load(audio_path)

# Calculate the duration of the audio in seconds
duration = len(audio) / sr

# Generate the time axis for the waveform plot
time = np.arange(0, duration, 1/sr)

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, audio, color='#f2b03b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.grid(True)
plt.show()

#########################################################################################

#loading names of columns in the dataset
hivedata.columns.tolist()

#Subsetting dataset
weather_columns = ['hive_temp', 
                    'hive_humidity', 
                    'hive_pressure', 
                    'weather_temp', 
                    'weather_humidity', 
                    'wind_speed', 
                    'cloud_coverage', 
                    'queen_status']

hivedata_weather = hivedata[weather_columns]

#Checking linear correlation
hivedata_corr = hivedata_weather.corr()
corr = hivedata_weather.corr()
corr_df = pd.DataFrame(corr)
#Karl Pearson's correlation coefficients will not be able to help
#since target variable is categorical.

#Visually exploring data using a pairs plot
#Checking for linear correlations
sns.pairplot(hivedata_weather)
plt.show()

# Defining the figure size
plt.figure(figsize=(12, 10))


# Boxplot for hive pressure and queen status
sns.boxplot(x='queen_status', 
            y='hive_pressure', 
            data=hivedata, 
            color='#f2b03b')

# Update the x-axis labels
plt.xticks([0, 1, 2, 3], 
           ['Original Queen',
            'Absent',
            'Present & Rejected',
            'Present & Newly Accepted'])

# Adding labels
plt.xlabel('Queen Status')
plt.ylabel('Hive Pressure')
plt.title('Hive Pressure by Queen Status')

# Showing the plot
plt.show()

#The plot is not helpful for thesis document.
