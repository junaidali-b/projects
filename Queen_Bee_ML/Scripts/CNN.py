#Importing required libraries for modelling gradient boosting models and 
#visualising results

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pandas as pd


#At this point, the the Spyder session has been populated with variables
#prepared in the past, using a pickle backup file. Warnings about variables not
#defined in the code folling this are to be ignored, as those variables will
#be contained in the .pkl backup file, and can be used directly from the
#environment.

#Loading custom pickle functions
#The relative path 'Scripts/Pickle Backup.py' will only work if you are in
#the correct working directory which contains the subfoler 'Scripts' and
#the file "Pickle Backup.py" within that subfolder
exec(open('Scripts/Pickle Backup.py').read())

#Declaring device to be used for modelling- GPU
tf.config.set_visible_devices([], 'GPU')

#Model 1:
#Using segmented spectrograms
#Optimizer- Adam with learning rate 0.02
#Sample size 64

# Split the dataset into train and test sets
train_data, test_data = train_test_split(hivedata_spec_long, 
                                         test_size=0.2, 
                                         random_state=110)

# Create an ImageDataGenerator for data augmentation and normalization
image_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
)

# Converting data type of labels to string
train_data['queen_status'] = train_data['queen_status'].astype(str)
test_data['queen_status'] = test_data['queen_status'].astype(str)
train_labels = train_data['queen_status']
test_labels = test_data['queen_status']

# Calculate class weights
weights_1 = compute_class_weight(class_weight ='balanced', 
                               classes = np.unique(train_labels), 
                               y = train_labels)
weights_1 = dict(enumerate(weights_1))

# Define the data generators for training and testing
batch_size = 64
train_generator = image_gen.flow_from_dataframe(
    train_data,
    x_col='spec_loc',
    y_col='queen_status',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224, 224)
)

test_generator_1 = image_gen.flow_from_dataframe(
    test_data,
    x_col='spec_loc',
    y_col='queen_status',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224, 224)
)

# Build the CNN model
model_1 = Sequential([
    
    #Convolutional layer 1
    Conv2D(16, (3, 3), 
           activation='relu', 
           input_shape=(224, 224, 3)),
    
    #Pooling layer 1
    MaxPooling2D(2, 2),
    
    #Convolutional later 2
    Conv2D(32, (3, 3), activation='relu'),
    
    #Pooling layer 2
    MaxPooling2D(2, 2),
    
    #Convolutional layer 3
    Conv2D(64, (3, 3), activation='relu'),
    
    #Pooling layer 3
    MaxPooling2D(2, 2),
    
    #Flattening images
    Flatten(),
    Dense(256, activation='relu'),
    
    #Output
    Dense(4, activation='softmax'),
])

# Compile the model
model_1.compile(optimizer=Adam(learning_rate = 0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Setting conditions for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model

cnn_history_1 = model_1.fit(
    train_generator,
    steps_per_epoch = len(train_data) // batch_size,
    epochs = 15,
    validation_data=test_generator_1,
    validation_steps=len(test_data) // batch_size,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss_cnn_1, cnn_acc_1 = model_1.evaluate(test_generator_1)
print(f"Test accuracy: {cnn_acc_1 * 100:.2f}%")

#Saving model
cnn_1_tf = model_1.save('D:\Python\Spyder\Honey Bee\Backups\CNN Models\cnn_model_1_tf', 
                        save_format = 'tf')

#Garbage collection (clearing RAM)
gc.collect()


#Model 2:
#Using merged spectrograms
#Optimizer- Adam with learning rate 0.001
#Sample size 32

#Declaring device to be used for modelling- GPU
tf.config.set_visible_devices([], 'GPU')

# Split the dataset into train and test sets
train_data, test_data = train_test_split(hivedata, 
                                         test_size=0.2, 
                                         random_state=110)

# Create an ImageDataGenerator for data augmentation and normalization
image_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
)

# Converting data type of labels to string
train_data['queen_status'] = train_data['queen_status'].astype(str)
test_data['queen_status'] = test_data['queen_status'].astype(str)
train_labels = train_data['queen_status']
test_labels = test_data['queen_status']
train_labels.value_counts()

# Calculate class weights
weights_2 = compute_class_weight(class_weight ='balanced', 
                               classes = np.unique(train_labels), 
                               y = train_labels)
weights_2 = dict(enumerate(weights_2))

# Define the data generators for training and testing
batch_size = 32
train_generator = image_gen.flow_from_dataframe(
    train_data,
    x_col='spec',
    y_col='queen_status',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224, 224)
)

test_generator_2 = image_gen.flow_from_dataframe(
    test_data,
    x_col='spec',
    y_col='queen_status',
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224, 224)
)

# Build the CNN model
model_2 = Sequential([
    
    #Convolutional layer 1
    Conv2D(16, (3, 3), 
           activation='relu', 
           input_shape=(224, 224, 3)),
    
    #Pooling layer 1
    MaxPooling2D(2, 2),
    
    #Convolutional later 2
    Conv2D(32, (3, 3), activation='relu'),
    
    #Pooling layer 2
    MaxPooling2D(2, 2),
    
    #Convolutional layer 3
    Conv2D(64, (3, 3), activation='relu'),
    
    #Pooling layer 3
    MaxPooling2D(2, 2),
    
    #Flattening images
    Flatten(),
    Dense(256, activation='relu'),
    
    #Output
    Dense(4, activation='softmax'),
])

# Compile the model
model_2.compile(optimizer=Adam(learning_rate = 0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Setting conditions for early stopping
early_stopping_2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model

cnn_history_2 = model_2.fit(
    train_generator,
    steps_per_epoch = len(train_data) // batch_size,
    epochs = 15,
    validation_data=test_generator_2,
    validation_steps=len(test_data) // batch_size,
    class_weight=weights_2,
    callbacks=[early_stopping_2]
)

# Evaluate the model on the test set
test_loss_cnn_2, cnn_acc_2 = model_2.evaluate(test_generator_2)
print(f"Test accuracy: {cnn_acc_2 * 100:.2f}%")

#Saving model
cnn_2_tf = model_2.save('D:\Python\Spyder\Honey Bee\Backups\CNN Models\cnn_model_2_tf', 
                        save_format = 'tf')

#Garbage collection (clearing RAM)
gc.collect()


#Model Diagnostic Plots

#Model Accuracy by Epoch

#Model 1

plt.plot(cnn_history_1.history['accuracy'])
plt.plot(cnn_history_1.history['val_accuracy'])
plt.title('Model 1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/cnn_acc_1.png")
plt.show()


#Model 2

plt.plot(cnn_history_2.history['accuracy'])
plt.plot(cnn_history_2.history['val_accuracy'])
plt.title('Model 2 accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/cnn_acc_2.png")
plt.show()


#Loss per Epoch

#Model 1

# Plot training & validation loss values
plt.plot(cnn_history_1.history['loss'])
plt.plot(cnn_history_1.history['val_loss'])
plt.title('Model 1 loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/cnn_loss_1.png")
plt.show()

#Model 2

# Plot training & validation loss values
plt.plot(cnn_history_2.history['loss'])
plt.plot(cnn_history_2.history['val_loss'])
plt.title('Model 2 loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/cnn_loss_2.png")
plt.show()


#Confusion matrix

#Model 1
cnn_y_pred_1 = model_1.predict(test_generator_1)
y_pred_classes = np.argmax(cnn_y_pred_1, axis=1)
y_true = test_generator_1.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Model 1 Confusion Matrix')

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/cnn_confusion_1.png")

plt.show()


#Model 2
cnn_y_pred_2 = model_2.predict(test_generator_2)
y_pred_classes_2 = np.argmax(cnn_y_pred_2, axis=1)
y_true = test_generator_2.classes

cm = confusion_matrix(y_true, y_pred_classes_2)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Model 2 Confusion Matrix')

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/cnn_confusion_2.png")

plt.show()


#Model Comparison

#Combining classification report for audio feature based models

# Assuming you have the classification reports already computed
report_cnn_1 = classification_report(test_generator_1.classes, 
                                     np.argmax(cnn_y_pred_1, axis=1),
                                     output_dict=True)
report_cnn_2 = classification_report(test_generator_2.classes, 
                                     np.argmax(cnn_y_pred_2, axis=1),
                                     output_dict=True)

# Extract metrics of class '2' from each classification report
metrics_class_2_cnn_1 = report_cnn_1['2']
metrics_class_2_cnn_2 = report_cnn_2['2']

# Create a DataFrame with the metrics
diag_cnn = pd.DataFrame({'Metric': metrics_class_2_cnn_1.keys(),
                         'Segmented Audio': [round(value, 2) for value in metrics_class_2_cnn_1.values()],
                         'Merged Audio': [round(value, 2) for value in metrics_class_2_cnn_2.values()]})

diag_cnn

#Updating pickle backup file
backup_env(audio_folder,
           hivedata,
           merged_audio_folder,
           merged_spec_folder,
           NaN_count,
           noaudio_data,
           weather_columns,
           spectrogram_columns,
           spectrogram_folder,
           subfolder_names,
           hivedata_weather,
           hivedata_spec_long,
           au_feat_col,
           hivedata_au_feat,
           svm_1_lin_acc,
           svm_1_rbf_acc,
           svm_1_poly_acc,
           svm_1_sig_acc,
           svm_2_acc,
           svm_3_acc,
           svm_4_acc,
           svm_1_lin_y_pred,
           svm_y_pred_2,
           svm_y_pred_3,
           svm_y_pred_4,
           svm_spec_y_test_2,
           rf_au_20_acc,
           rf_au_50_acc,
           rf_wea_20_acc,
           rf_wea_50_acc,
           rf_wea_20_y_pred,
           rf_wea_50_y_pred,
           rf_au_20_y_pred,
           rf_au_50_y_pred,
           hivedata_wea_spec,
           xgb_acc_1,
           xgb_y_pred_1,
           xgb_acc_2,
           xgb_y_pred_2,
           xgb_acc_3,
           xgb_y_pred_3,
           test_loss_cnn_1, 
           cnn_acc_1,
           cnn_y_pred_1,
           cnn_history_1,
           cnn_acc_2,
           cnn_y_pred_2,
           cnn_history_2,
           svm_1_all_acc,
           svm_metrics,
           rf_metrics,
           rf_feat_imp,
           xgb_report,
           xgb_metrics,
           xgb_3_feat_imp)

