#Loading required libraries

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from PIL import Image
import gc
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

#At this point, the the Spyder session has been populated with variables
#prepared in the past, using a pickle backup file. Warnings about variables not
#defined in the code folling this are to be ignored, as those variables will
#be contained in the .pkl backup file, and can be used directly from the
#environment.

#Model 1- Audio Features

#Defining predictors and target variable
x = hivedata_au_feat.iloc[:,:-1]
y = hivedata_au_feat.iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_1 = train_test_split(x,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=110)

#Defining weights for class imbalance
queen_status_weights = compute_class_weight(class_weight = 'balanced',
                                            classes = np.unique(y_train), 
                                            y = y_train)

queen_status_weights = dict(enumerate(queen_status_weights))


#Linear Kernel

#Training SVM Model
svm = SVC(kernel='linear', class_weight = queen_status_weights)
svm.fit(x_train, y_train)

#Predicting based on the training set
svm_1_lin_y_pred = svm.predict(x_test)

#Computing model accuracy
svm_1_lin_acc = round(metrics.accuracy_score(y_test_1, svm_1_lin_y_pred), 2)

#Garbage collection (clearing RAM)
gc.collect()

svm_1_lin_acc

#RBF Kernel

#Training SVM Model
svm = SVC(kernel='rbf', class_weight = queen_status_weights)
svm.fit(x_train, y_train)

#Predicting based on the training set
svm_1_rbf_y_pred = svm.predict(x_test)

#Computing model accuracy
svm_1_rbf_acc = round(metrics.accuracy_score(y_test_1, svm_1_rbf_y_pred), 2)

#Garbage collection (clearing RAM)
gc.collect()

svm_1_rbf_acc

#Polynomial Kernel

#Training SVM Model
svm = SVC(kernel='poly', class_weight = queen_status_weights)
svm.fit(x_train, y_train)

#Predicting based on the training set
svm_1_poly_y_pred = svm.predict(x_test)

#Computing model accuracy
svm_1_poly_acc = round(metrics.accuracy_score(y_test_1, svm_1_poly_y_pred), 2)

#Garbage collection (clearing RAM)
gc.collect()

svm_1_poly_acc

#Sigmoid Kernel

#Training SVM Model
svm = SVC(kernel='sigmoid', class_weight = queen_status_weights)
svm.fit(x_train, y_train)

#Predicting based on the training set
svm_1_sig_y_pred = svm.predict(x_test)

#Computing model accuracy
svm_1_sig_acc = round(metrics.accuracy_score(y_test_1, svm_1_sig_y_pred), 2)

#Garbage collection (clearing RAM)
gc.collect()

svm_1_sig_acc

#Comparing kernels
svm_1_all_acc = pd.DataFrame({'Kernel':['Linear', 
                                        'RBF', 
                                        'Polynomial', 
                                        'Sigmoid'],
                              'Accuracy':[svm_1_lin_acc, 
                                          svm_1_rbf_acc, 
                                          svm_1_poly_acc, 
                                          svm_1_sig_acc]})

svm_1_all_acc


#Model 2- Based on Spectrograms

# Constants for batch processing
batch = 50  # Adjust this value based on your available memory
image_size = (224, 224)  # Adjust the desired size for the spectrograms
flattened_specs = "D:/Python/Spyder/Honey Bee/Data/flattened_specs"

# Create the directory for processed data if it doesn't exist
if not os.path.exists(flattened_specs):
    os.makedirs(flattened_specs)

# Function to process a batch of spectrograms
def process_batch(batch_specs, batch_labels):
    batch_flattened_specs = []
    
    for spec_path, label in zip(batch_specs, batch_labels):
        processed_spec_path = os.path.join(flattened_specs, os.path.basename(spec_path))
        
        if not os.path.exists(processed_spec_path):
            loaded_spec = Image.open(spec_path)
            resized_spec = loaded_spec.resize(image_size)  # Resize the image
            resized_spec.save(processed_spec_path)  # Save the processed spectrogram
            
        else:
            resized_spec = Image.open(processed_spec_path)
        spec_array = np.array(resized_spec, dtype='uint8')
        flattened_spec = spec_array.flatten()
        batch_flattened_specs.append(flattened_spec)
        
    return batch_flattened_specs

# Split the data into training and testing sets in batches
specs_train = []
labels_train = []
specs_test = []
labels_test = []

num_batches = int(np.ceil(len(hivedata) / batch))
for i in range(num_batches):
    start_idx = i * batch
    end_idx = (i + 1) * batch
    batch_specs = hivedata['spec'][start_idx:end_idx]
    batch_labels = hivedata['queen_status'][start_idx:end_idx]
    batch_flattened_specs = process_batch(batch_specs, batch_labels)
    
    # Split the batch into training and testing sets
    x_train_batch, x_test_batch, y_train_batch, y_test_batch = train_test_split(batch_flattened_specs, 
                                                                                batch_labels,
                                                                                test_size=0.2, 
                                                                                random_state=110)
    
    # Extend the training and testing sets with the batch data
    specs_train.extend(x_train_batch)
    labels_train.extend(y_train_batch)
    specs_test.extend(x_test_batch)
    labels_test.extend(y_test_batch)
    
    gc.collect()

# Convert the data into numpy arrays
#'uint8' has to be added to avoid errors

x_train = np.array(specs_train, dtype='uint8')
y_train = np.array(labels_train)
x_test = np.array(specs_test, dtype='uint8')
svm_spec_y_test_2 = np.array(labels_test)

# Train the SVM Model
svm = SVC(kernel='linear', class_weight = 'balanced')
svm.fit(x_train, y_train)

# Make Predictions
svm_y_pred_2 = svm.predict(x_test)

# Evaluate the Model
svm_2_acc = metrics.accuracy_score(svm_spec_y_test_2, svm_y_pred_2)
svm_2_acc

#Garbage collector (frees up RAM)
gc.collect()


#Model 3- Weather Data

#Defining predictors and target variable
x = hivedata_weather.fillna(method = 'ffill').iloc[:,:-1]
y = hivedata_weather.fillna(method = 'ffill').iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_3 = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=110)

#Linear Kernel

#Training SVM Model
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

#Predicting based on the training set
svm_y_pred_3 = svm.predict(x_test)

#Computing model accuracy
svm_3_acc = round(metrics.accuracy_score(y_test_3, svm_y_pred_3), 2)

#Garbage collection (clearing RAM)
gc.collect()

svm_3_acc


#Model 4- Weather and Audio Data Combined

#Defining predictors and target variable
x = hivedata_wea_spec.fillna(method = 'ffill').iloc[:,:-1]
y = hivedata_wea_spec.fillna(method = 'ffill').iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_4 = train_test_split(x, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=110)

#Linear Kernel

#Training SVM Model
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

#Predicting based on the training set
svm_y_pred_4 = svm.predict(x_test)

#Computing model accuracy
svm_4_acc = round(metrics.accuracy_score(y_test_4, svm_y_pred_4), 2)

#Garbage collection (clearing RAM)
gc.collect()

#Computing feature importance

# Get the coefficients of the support vectors for linear kernel SVM
svm_4_coef = svm.coef_

# Calculate the absolute sum of coefficients for each feature
svm_4_imp_scores = abs(svm_4_coef).sum(axis=0)

# Create a dictionary to associate feature names with their importance scores
svm_4_feat_imp = dict(zip(x.columns, svm_4_imp_scores))

# Print or display the feature importance scores
print(svm_4_feat_imp)

svm_4_acc


#Creating summary of model diagnostics with class '2'- Queen Present & Rejected
#Combining Classification Reports

# Assuming you have the classification reports already computed
svm_report_1 = classification_report(y_test_1, svm_1_lin_y_pred, output_dict=True)
svm_report_2 = classification_report(svm_spec_y_test_2, svm_y_pred_2, output_dict=True)
svm_report_3 = classification_report(y_test_3, svm_y_pred_3, output_dict=True)
svm_report_4 = classification_report(y_test_4, svm_y_pred_4, output_dict=True)

# Extract metrics of class '2' from each classification report
metrics_class_2_svm_1 = svm_report_1['2']
metrics_class_2_svm_2 = svm_report_2['2']
metrics_class_2_svm_3 = svm_report_3['2']
metrics_class_2_svm_4 = svm_report_4['2']

svm_metrics = pd.DataFrame({
    'Metric': list(metrics_class_2_svm_1.keys()) + ['model accuracy'],
    'Audio Features': [round(value, 2) for value in list(metrics_class_2_svm_1.values()) + [svm_1_sig_acc]],
    'Spectrograms': [round(value, 2) for value in list(metrics_class_2_svm_2.values()) + [svm_2_acc]],
    'Weather': [round(value, 2) for value in list(metrics_class_2_svm_3.values()) + [svm_3_acc]],
    'Audio & Weather': [round(value, 2) for value in list(metrics_class_2_svm_4.values()) + [svm_4_acc]]
})

# Printing table without 'support' row
print(svm_metrics.drop(2))

#Updating pickle backup
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
           svm_1_all_acc,
           svm_metrics)