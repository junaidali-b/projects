#Importing libraries required for UMAP
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import umap
import matplotlib.colors as mcolours
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#At this point, the the Spyder session has been populated with variables
#prepared in the past, using a pickle backup file. Warnings about variables not
#defined in the code folling this are to be ignored, as those variables will
#be contained in the .pkl backup file, and can be used directly from the
#environment.

#2D UMAP

# Load and process the spectrogram images
spectrograms = []
specs = hivedata['spec'].values

# Iterate over the file locations and load the spectrogram images
for file_location in specs:
    if isinstance(file_location, str):
        spectrogram = cv2.imread(str(file_location), 0)
        spectrograms.append(spectrogram)

# Convert the list of images into a single numpy array
x = np.array(spectrograms)

# Reshape the array to have a single channel (grayscale)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

# Get the target variable (Queen Status)
y = hivedata['queen_status'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=110)

# Flatten the image data
X_train_flattened = X_train.reshape(X_train.shape[0], -1)

# Apply UMAP for dimensionality reduction (2D)
reducer = umap.UMAP(n_components=2, random_state=110)
X_train_umap = reducer.fit_transform(X_train_flattened)

#Colour map
#'rgby' stands for r- red, g- green, b- blue, y- yellow
cmap_rgby = mcolours.ListedColormap(['red', 'green', 'blue', 'yellow'])

# Create a scatter plot of the UMAP visualization
umap_2d = plt.scatter(X_train_umap[:, 0], 
                      X_train_umap[:, 1], 
                      c=y_train, 
                      cmap=cmap_rgby)

legend = plt.colorbar(umap_2d, ticks=[0, 1, 2, 3])
legend.set_label('Queen Status')
legend.set_ticklabels(['Original Queen', 
                       'Absent', 
                       'Present & Rejected', 
                       'Present & Newly Accepted'])

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization of Spectrograms with Queen Status')

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/2d_umap.png")
plt.show()


#3D UMAP

# Apply UMAP for dimensionality reduction (3D)
reducer_3d = umap.UMAP(n_components=3, random_state=110)
X_train_umap_3d = reducer_3d.fit_transform(X_train_umap)

# Create a 3D scatter plot of the UMAP visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
umap_3d = ax.scatter(X_train_umap_3d[:, 0], 
                     X_train_umap_3d[:, 1], 
                     X_train_umap_3d[:, 2], 
                     c=y_train, 
                     cmap=cmap_rgby)

# Creating discrete legend
legend = ax.legend(*umap_3d.legend_elements(), 
                   title="Queen Status", 
                   loc='upper left')
ax.add_artist(legend)

ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')
ax.set_title('UMAP Visualization of Spectrograms with Queen Status (3D)')
ax.legend()
#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/3d_umap.png")
plt.show()

#Model Diagnostics

#Silhouette score

#For 2D UMAP Model
silhouette_umap_2d = silhouette_score(X_train_umap, y_train)
print("Silhouette Score (2D UMAP):", silhouette_umap_2d)

#For 3D UMAP Model
silhouette_umap_3d = silhouette_score(X_train_umap_3d, y_train)
print("Silhouette Score (3D UMAP):", silhouette_umap_3d)