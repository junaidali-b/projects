#Importing required libraries
import os
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import datetime
from pydub import AudioSegment


#Loading CSV data
hivedata = pd.read_csv("D:/Python/Spyder/Honey Bee/Data/all_data_updated.csv")

#Removing spaces from column names
hivedata.columns = hivedata.columns.str.replace(' ', '_')

#Defining function to segment all audio files into specific number of batches.
#Each batch of audio files will be stored in its own subfolder within the
#main directory.

def subfolders_by_count(directory, max_files):
    file_list = os.listdir(directory)
    total_files = len(file_list)
    num_subfolders = math.ceil(total_files / max_files)
    
    error_files = []  # Variable to store filenames causing permission errors
    
    for i in range(num_subfolders):
        start_index = i * max_files
        end_index = (i + 1) * max_files
        subfolder_path = os.path.join(directory, str(i + 1))
        os.makedirs(subfolder_path)
        
        for file_name in file_list[start_index:end_index]:
            file_path = os.path.join(directory, file_name)
            destination_path = os.path.join(subfolder_path, file_name)
            
            try:
                os.rename(file_path, destination_path)
            except PermissionError:
                error_files.append(file_name)
    
    return error_files

#Processing segments of the audio first
#Defining audio main folder location and maximum number of files per subfolder
audio_folder = 'D:/Python/Spyder/Honey Bee/Data/audio_files'
max_files = 150

#Using defined function to create batches for audio file segments in the local
#machine
subfolders_by_count(audio_folder, max_files)

#Creating new columns for segmented audio file locations
hivedata['audio_0'] = pd.Series(dtype = 'object')
hivedata['audio_1'] = pd.Series(dtype = 'object')
hivedata['audio_2'] = pd.Series(dtype = 'object')
hivedata['audio_3'] = pd.Series(dtype = 'object')
hivedata['audio_4'] = pd.Series(dtype = 'object')
hivedata['audio_5'] = pd.Series(dtype = 'object')

#Matching names of audio files segments with file names in the hive data CSV

#Iterating over each row in the DataFrame
for index, row in hivedata.iterrows():
    
    # Column containing the file names
    file_name = os.path.splitext(row['file_name'])[0]  
    
    # Iterate over the segment numbers
    for segment_num in range(0, 6):  # Segments 0 to 5
        
        # Construct the segment file name
        segment_file_name = f"{file_name}__segment{segment_num}.wav"
        
        # Search for the file in the subfolders
        for subfolder_name in os.listdir(audio_folder):
            subfolder_path = os.path.join(audio_folder, subfolder_name)
            
            if os.path.isdir(subfolder_path):
                file_path = os.path.join(subfolder_path, segment_file_name)
                
                if os.path.isfile(file_path):
                    #If file is found storing the file location
                    #in the corresponding column
                    column_name = f"audio_{segment_num}"  
                    hivedata.loc[index, column_name] = file_path.replace("\\", "/")
                    break


#Merging Audio file segments for each row into bigger audio files

merged_audio_folder = 'D:/Python/Spyder/Honey Bee/Data/audio_files/merged'
if not os.path.exists(merged_audio_folder):
    os.makedirs(merged_audio_folder)

# Iterate over each row in the DataFrame
for index, row in hivedata.iterrows():
    # List to store the audio data of all segments
    merged_audio_data = []
    
    # Iterate over the segment numbers
    for segment_num in range(0, 6):
        # Get the segment file path
        segment_file_path = row[f'audio_{segment_num}']
        
        # Check if the segment file path is not NaN
        if isinstance(segment_file_path, str):
            # Read the segment audio data
            segment_audio_data, sample_rate = sf.read(segment_file_path)
            
            # Append the segment audio data to the merged audio data list
            merged_audio_data.append(segment_audio_data)
    
    # Check if there is any audio data to merge
    if merged_audio_data:
        # Concatenate the audio data from all segments in sequence
        merged_audio_data = np.concatenate(merged_audio_data)
        
        # Get the filename for the merged audio file
        original_file_name = os.path.splitext(row['file_name'])[0]
        merged_audio_file_name = f"{original_file_name}.wav"
        merged_audio_file_path = os.path.join(merged_audio_folder, merged_audio_file_name)
        
        # Write the merged audio data to the merged audio file
        sf.write(merged_audio_file_path, merged_audio_data, sample_rate)
        
        # Update the 'segment_count' column
        segment_count = sum(not pd.isna(row[f'audio_{segment_num}']) for segment_num in range(0, 6))
        hivedata.loc[index, 'segment_count'] = segment_count

#Dividing files in main merged audio folder into batches of 50 files each

max_files = 50
subfolders_by_count(merged_audio_folder, max_files) #using the predefined function
        
#Searching for and recording the file locations of merged audio files for
#each row

# Create the 'audio' column
hivedata['audio'] = np.nan

#Matching names of merged audio files with file names in the hive data CSV
#This is to record the locations of the audio files on present on the local
#machine, in the CSV for each row.

# Iterate over each row in the DataFrame
for index, row in hivedata.iterrows():
    # Column containing the file names
    file_name = os.path.splitext(row['file_name'])[0]
    
    # Construct the merged audio file name
    merged_audio_file_name = f"{file_name}.wav"
    
    # Search for the merged audio file in the subfolders
    for subfolder_name in os.listdir(merged_audio_folder):
        subfolder_path = os.path.join(merged_audio_folder, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            file_path = os.path.join(subfolder_path, merged_audio_file_name)
            
            if os.path.isfile(file_path):
                #If file is found storing the file location
                #in the corresponding column
                hivedata.loc[index, 'audio'] = file_path.replace("\\", "/")
                break


#Checking for rows containing no audio file locations (NaN)
NaN_count = sum(hivedata['audio'].isna())
NaN_count

#Collecting rows where there are not even one audio file locations found
noaudio_data = hivedata[hivedata['audio'].isna() == True]           
noaudio_data.head()

#Dropping rows where there is no audio present
hivedata = hivedata.dropna(subset=['audio'])

#Processing Spectrograms

# Creating subfolders for spectrograms with the same names as the audio folder
spectrogram_folder = 'D:/Python/Spyder/Honey Bee/Data/spec_files'

# Getting the list of subfolders in the audio folder
subfolder_names = os.listdir(audio_folder)

# Creating corresponding subfolders in the spectrogram main folder
for subfolder_name in subfolder_names:
    subfolder_path = os.path.join(spectrogram_folder, subfolder_name)
    os.makedirs(subfolder_path)
    
def create_spectrogram(audio_folder, spectrogram_folder, audio_subfolder, spectrogram_subfolder):
    audio_subfolder_path = os.path.join(audio_folder, str(audio_subfolder))
    spectrogram_subfolder_path = os.path.join(spectrogram_folder, str(spectrogram_subfolder))
    
    audio_files = os.listdir(audio_subfolder_path)

    for audio_file in audio_files:
        audio_path = os.path.join(audio_subfolder_path, audio_file)
        spectrogram_file = os.path.splitext(audio_file)[0] + '.png'
        spectrogram_path = os.path.join(spectrogram_subfolder_path, spectrogram_file)

        # Load audio file
        audio, sr = librosa.load(audio_path)

        # Compute spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, power=2.0)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Resize spectrogram to a fixed size
        spectrogram = np.resize(spectrogram, (spectrogram.shape[0], 224))

        # Normalize spectrogram
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

        # Save spectrogram as an image
        plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(spectrogram, cmap='inferno', origin='lower', aspect='auto')
        plt.axis('off')

        # Replace backslashes with forward slashes in the spectrogram path
        spectrogram_path = spectrogram_path.replace("\\", "/")

        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()


# Using function in a loop to produce spectrograms for all subfolders, in 2
# batches. Subfolders have been created to have ready segments in case heavier
# computations of the spectrograms are required, or if more detailed
# spectrograms are to be produced. In case the compiler runs into memory based
# issues, the segments can help in continuing where it left off instead of
# starting over again.

#Batch 1
for i in range(1, 25):
    create_spectrogram(audio_folder = audio_folder,
                       spectrogram_folder = spectrogram_folder,
                       audio_subfolder=i, 
                       spectrogram_subfolder=i)

#It is recommended to restart the IDE after batch 1, in case the computer is
#unable to handle too much data at once (that is, cases where the RAM is 
#very low)

#Batch 2
for i in range(25, 49):
    create_spectrogram(audio_folder = audio_folder,
                       spectrogram_folder = spectrogram_folder,
                       audio_subfolder=i, 
                       spectrogram_subfolder=i)

#Creating new columns for spectrograms of segmented audio file locations
hivedata['spec_0'] = pd.Series(dtype = 'object')
hivedata['spec_1'] = pd.Series(dtype = 'object')
hivedata['spec_2'] = pd.Series(dtype = 'object')
hivedata['spec_3'] = pd.Series(dtype = 'object')
hivedata['spec_4'] = pd.Series(dtype = 'object')
hivedata['spec_5'] = pd.Series(dtype = 'object')


#Creating spectrograms for merged audio files

# Creating subfolders for spectrograms with the same names as the 
# merged audio folder

merged_spec_folder = 'D:/Python/Spyder/Honey Bee/Data/spec_files/merged'

# Getting the list of subfolders in the audio folder
subfolder_names = os.listdir(merged_audio_folder)

# Creating corresponding subfolders in the spectrogram main folder
for subfolder_name in subfolder_names:
    subfolder_path = os.path.join(spectrogram_folder, subfolder_name)
    os.makedirs(subfolder_path)
    
#Similar to the segmented audio files this too needs to be done in batches

#Batch 1
for i in range(1, 13):
    create_spectrogram(audio_folder = merged_audio_folder,
                       spectrogram_folder = merged_spec_folder,
                       audio_subfolder=i, 
                       spectrogram_subfolder=i)

#Batch 2
for i in range(13, 26):
    create_spectrogram(audio_folder = merged_audio_folder,
                       spectrogram_folder = merged_spec_folder,
                       audio_subfolder=i, 
                       spectrogram_subfolder=i)

# Create the column for spectrograms of merged audio clips (spec)
hivedata['spec'] = np.nan

#Searching for and recording the file locations of spectrograms (from merged 
#audio) for each row

# Iterate over each row in the DataFrame
for index, row in hivedata.iterrows():
    # Column containing the file names
    file_name = os.path.splitext(row['file_name'])[0]
    
    # Construct the merged audio file name
    merged_spec_file_name = f"{file_name}.png"
    
    # Search for the merged audio file in the subfolders
    for subfolder_name in os.listdir(merged_spec_folder):
        subfolder_path = os.path.join(merged_spec_folder, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            file_path = os.path.join(subfolder_path, merged_spec_file_name)
            
            if os.path.isfile(file_path):
                # If the merged audio file is found,
                #the file location is stored in the corresponding column
                hivedata.loc[index, 'spec'] = file_path.replace("\\", "/")
                break

#Matching names of spectrograms of audio files segments with file names 
#in the hive data CSV

# Iterate over each row in the DataFrame
for index, row in hivedata.iterrows():
    
    # Column containing the file names
    file_name = os.path.splitext(row['file_name'])[0]  
    
    # Iterate over the segment numbers
    for segment_num in range(0, 6):  # Segments 0 to 5
        
        # Construct the segment file name
        segment_file_name = f"{file_name}__segment{segment_num}.png"
        
        # Search for the file in the subfolders
        for subfolder_name in os.listdir(spectrogram_folder):
            subfolder_path = os.path.join(spectrogram_folder, subfolder_name)
            
            if os.path.isdir(subfolder_path):
                file_path = os.path.join(subfolder_path, segment_file_name)
                
                if os.path.isfile(file_path):
                    # File found, store the file location in the corresponding column
                    column_name = f"spec_{segment_num}"  
                    hivedata.loc[index, column_name] = file_path.replace("\\", "/")
                    break

#Loading custom pickle functions
#The relative path 'Scripts/Pickle Backup.py' will only work if you are in
#the correct working directory which contains the subfoler 'Scripts' and
#the file "Pickle Backup.py" within that subfolder
exec(open('Scripts/Pickle Backup.py').read())

#Backing up environment
backup_env(audio_folder,
           hivedata,
           image_paths,
           merged_audio_folder,
           merged_spec_folder,
           NaN_count,
           noaudio_data,
           weather_columns,
           spectrogram_columns,
           spectrogram_folder,
           subfolder_names,
           hivedata_weather)

#Extracting audio features and storing it in the hivedata dataset

#Creating function to extract audio features

def extract_audio_features(path):
    # Load audio file
    au, sr = librosa.load(path)
    
    # Extract features
    features = {}
    
    # Time-domain features
    features['duration'] = len(au) / sr
    
    # Frequency-domain features
    stft = np.abs(librosa.stft(au))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)[0])
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0])
    features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr)[0])
    features['chroma_stft'] = librosa.feature.chroma_stft(S=stft, sr=sr)
    features['chroma_cqt'] = librosa.feature.chroma_cqt(y=au, sr=sr)
    features['chroma_cens'] = librosa.feature.chroma_cens(y=au, sr=sr)
    features['mfcc'] = librosa.feature.mfcc(y=au, sr=sr)
    features['rmse'] = np.mean(librosa.feature.rms(y=au))
    
    # Additional features
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(au))
    features['tempogram'] = librosa.feature.tempogram(y=au, sr=sr)
    
    return features

# Iterate over each row in the DataFrame
for index, row in hivedata.iterrows():
    # Extract audio features
    audio_file = row['audio']
    audio_features = extract_audio_features(audio_file)
    
    # Update the DataFrame with the audio features
    for feature, feature_values in audio_features.items():
        column_name = f'{feature}'
        hivedata.loc[index, column_name] = np.mean(feature_values)
        
        
#Backing up environment (second time)
backup_env(audio_folder,
           hivedata,
           image_paths,
           merged_audio_folder,
           merged_spec_folder,
           NaN_count,
           noaudio_data,
           weather_columns,
           spectrogram_columns,
           spectrogram_folder,
           subfolder_names,
           hivedata_weather)


#Computing decibels

def audio_decibels(path_i):
    audio_i, _ = librosa.load(path_i, sr = None)
    decibels_i = librosa.amplitude_to_db(librosa.feature.rms(y=audio_i), ref=0.1)
    return decibels_i

hivedata['decibels'] = hivedata['audio'].apply(audio_decibels)

#Correcting format of date and time column to date-time object
hivedata['date'] = pd.to_datetime(hivedata['date'])

#Classifying rows into seasons based on dates
#Location used for reference is California, since the data was collected from
#there

#Custom fuction to classify seasons for Califonia
def cali_seasons(month):
    
    #Spring months- March, April, May
    if month in [3, 4, 5]:
        return 'Spring'
    
    #Summer months- June, July, August
    elif month in [6, 7, 8]:
        return 'Summer'
    
    #Autumn months- September October November
    elif month in [9, 10, 11]:
        return 'Autumn'
    
    #Winter months- December, January, February
    else:
        return 'Winter'
    
#Classifying rows into seasons based on the defined function
#The month part of the date is to be used as input
hivedata['season'] = hivedata['date'].dt.month.apply(cali_seasons)

#Checking unique values
#hivedata['season'].value_counts() #Only summer data present 

#Categorising time of the day into Day and Night
#Based on day timings on https://www.timeanddate.com for Los Angeles California

#Sunrise and Sunset times as observed on the website for June 2022 for LA
#The sunrise and sunset timings for entire Summer has been assumed to be the same
#The dataset only contains data related to Summer as observed in the previous step

sunrise = datetime.time(5, 40, 0) #05:40 hours
sunset = datetime.time(20, 0, 0) #20:00 hours

#Function to classify rows into day or night based on sunrise and sunset
#cut-off timings

def cali_daynight(time):
    #Sunrise time is included in 'Day'
    #Sunset time is included in 'Night'
    if sunrise <= time < sunset:
        return "Day"
    else:
        return "Night"
    
#Using function to create column for day and night
hivedata['daytime'] = hivedata['date'].dt.time.apply(cali_daynight)

#Checking unique values
#hivedata['daytime'].value_counts()

#Creating dataset only containing spectrogram file locations and labels

#Merging columns for spectrograms of audio segments

hivedata_spec_long = pd.melt(hivedata,
                             id_vars=['queen_status'],
                             value_vars=spectrogram_columns,
                             var_name='spec_segment',
                             value_name='spec_loc')

#Reordering columns and removing unwanted columns
hivedata_spec_long = hivedata_spec_long[['spec_loc', 'queen_status']]

#Removing NaN
hivedata_spec_long = hivedata_spec_long.dropna(subset=['spec_loc'])


# 1 second audio dataset for CNN

#Folder to store 1 second audio audio files
audio_folder_1sec = "D:/Python/Spyder/Honey Bee/Data/audio_files/audio_by_sec"

# Create an empty list to hold the segment information
segment_info_list = []

# Loop through each row in hivedata
for index, row in hivedata.iterrows():
    audio_path = row['audio']
    queen_status = row['queen_status']
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_path)
    
    # Duration of each segment (in milliseconds)
    segment_duration = 1000  # 1 second
    
    # Iterate through the audio in segments
    for i in range(0, len(audio), segment_duration):
        # Get the current segment
        segment = audio[i:i + segment_duration]
        
        # Create a filename for the segment
        segment_filename = f"{index}_segment_{i // segment_duration}.wav"
        segment_path = os.path.join(audio_folder_1sec, segment_filename)
        
        # Save the segment to the output directory
        segment.export(segment_path, format="wav")
        
        # Create a dictionary for the segment information and append it to the list
        segment_info = {
            'segment_path': segment_path,
            'parent_audio': audio_path,
            'queen_status': queen_status
        }
        segment_info_list.append(segment_info)

# Create a DataFrame from the list of segment_info
hivedata_audio_1sec = pd.DataFrame(segment_info_list)

#Creating spectrograms for 1 second audio clips
spec_folder_1sec = "D:\Python\Spyder\Honey Bee\Data\spec_files\spec_by_sec"

# Loop through each row in the DataFrame
for index, row in hivedata_audio_1sec.iterrows():
    segment_path = row['segment_path']
    
    # Load the audio file
    audio, sr = librosa.load(segment_path)
    
    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, power=2.0)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Resize spectrogram to a fixed size
    spectrogram = np.resize(spectrogram, (spectrogram.shape[0], 224))
    
    # Normalize spectrogram
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    
    # Create the spectrogram filename
    spectrogram_file = f"{index}_segment_spec_{i // segment_duration}.png"
    spectrogram_path = os.path.join(spec_folder_1sec, spectrogram_file)
    
    # Replace backslashes with forward slashes in the spectrogram path
    spectrogram_path = spectrogram_path.replace("\\", "/")
    
    # Save the spectrogram as an image
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(spectrogram, cmap='inferno', origin='lower', aspect='auto')
    plt.axis('off')
    plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Update the 'spec' column with the spectrogram path
    hivedata_audio_1sec.at[index, 'spec'] = spectrogram_path

#Counting NaN
#sum(hivedata_spec_long['spec_loc'].isna())

#Updating pickle backup
#Backing up environment (second time)
backup_env(audio_folder,
           hivedata,
           image_paths,
           merged_audio_folder,
           merged_spec_folder,
           NaN_count,
           noaudio_data,
           weather_columns,
           spectrogram_columns,
           spectrogram_folder,
           subfolder_names,
           hivedata_weather,
           hivedata_spec_long)