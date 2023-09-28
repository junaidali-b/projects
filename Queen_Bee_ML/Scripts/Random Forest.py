#Importing required libraries for modelling random forest and visualising
#results

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics
import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import pandas as pd

#At this point, the the Spyder session has been populated with variables
#prepared in the past, using a pickle backup file. Warnings about variables not
#defined in the code folling this are to be ignored, as those variables will
#be contained in the .pkl backup file, and can be used directly from the
#environment.

#Modelling

#Random Forest models for weather data

#Defining predictor and target variables
x_wea = hivedata_weather.iloc[:, :-1].values
y_wea = hivedata_weather.iloc[:, -1].values

#Splitting data into training and testing sets
#80% for training and 20% for testing
#Seed used: 110
x_wea_train, x_wea_test, y_wea_train, y_wea_test = train_test_split(x_wea,
                                                                    y_wea,
                                                                    test_size=0.2,
                                                                    train_size=0.8,
                                                                    random_state=110)

#Defining weights
#Defining weights for class imbalance
queen_status_weights = compute_class_weight(class_weight = 'balanced',
                                            classes = np.unique(y_wea_train), 
                                            y = y_wea_train)

queen_status_weights = dict(enumerate(queen_status_weights))
 
#Clearing memory
gc.collect()

#Creating a random forest model with maximum depth 10
#maximum sample size is set to 20%
#Since the dataset is huge, 5 trees with 20% of the test data in each suffices
rf_wea_20 = RandomForestClassifier(max_depth= 9,
                                       max_samples= 0.2,
                                       max_features= 3,
                                       n_estimators = 20,
                                       warm_start=True,
                                       class_weight=queen_status_weights)

#Defining imputer to impute missing values in the numeric columns using medians
imp_median = SimpleImputer(strategy = 'median')

#Creating pipeline for the imputer to function along with the random forest model
rf__weather_pipe_20 = make_pipeline(imp_median, rf_wea_20)

#Fitting the model
rf__weather_pipe_20.fit(x_wea_train, y_wea_train)
#Clearing memory
gc.collect()

#Using the model on the testing set
rf_wea_20_y_pred = rf_wea_20.predict(x_wea_test)

#Checking the model accuracy
rf_wea_20_acc = metrics.accuracy_score(y_wea_test, rf_wea_20_y_pred)
print(rf_wea_20_acc)

#Clearing memory
gc.collect()

#Attempting to use a heavier model on the weather data
rf_wea_50 = RandomForestClassifier(max_depth= 9, 
                               max_samples= 0.5,
                               max_features= 5,
                               n_estimators = 50,
                               warm_start=True,
                               class_weight=queen_status_weights)

#Building pipeline for imputer to work with rf_wea_50
rf_wea_pipe_50 = make_pipeline(imp_median, rf_wea_50)

#Fitting the model
rf_wea_pipe_50.fit(x_wea_train, y_wea_train)
#Clearing memory
gc.collect()

#Using the model on the testing set
rf_wea_50_y_pred = rf_wea_pipe_50.predict(x_wea_test)

#Checking and storing the model accuracy
rf_wea_50_acc = metrics.accuracy_score(y_wea_test, rf_wea_50_y_pred)
print(rf_wea_50_acc)

#Clearing memory
gc.collect()

#Random Forest models for audio feature data

#Series of column names related to audio features
au_feat_col = ['spectral_centroid',
               'spectral_bandwidth',
               'spectral_contrast',
               'spectral_rolloff',
               'chroma_stft',
               'chroma_cqt',
               'chroma_cens',
               'mfcc',
               'rmse',
               'zero_crossing_rate',
               'tempogram',
               'queen_status']

#Using series of column names as a mask to subset data
hivedata_au_feat = hivedata[au_feat_col]

#Defining predictors and target variable
x_au = hivedata_au_feat.iloc[:, :-1].values
y_au = hivedata_au_feat.iloc[:, -1].values

#Defining training and testing sets
x_au_train, x_au_test, y_au_train, y_au_test = train_test_split(x_au,
                                                                y_au,
                                                                test_size=0.2,
                                                                train_size=0.8,
                                                                random_state=110)

#Modifying rf_au_20 to suit audio features subset
rf_au_20 = RandomForestClassifier(max_depth= 9, 
                               max_samples= 0.2,
                               max_features= 5,
                               n_estimators = 20,
                               warm_start=True,
                               class_weight=queen_status_weights)

#Clearing memory
gc.collect()

#Fitting modified rf_au_20
rf_au_20.fit(x_au_train, y_au_train)

#Clearing memory
gc.collect()

#Using the model on the testing set
rf_au_20_y_pred = rf_au_20.predict(x_au_test)

#Checking the model accuracy
rf_au_20_acc = metrics.accuracy_score(y_au_test, rf_au_20_y_pred)
print(rf_au_20_acc)

#Clearing memory
gc.collect()


#Modifying rf_50 to suit audio features subset
rf_au_50 = RandomForestClassifier(max_depth= 9, 
                               max_samples= 0.5,
                               max_features= 8,
                               n_estimators = 50,
                               warm_start=True,
                               class_weight=queen_status_weights)

#Clearing memory
gc.collect()

#Fitting modified rf_20
rf_au_50.fit(x_au_train, y_au_train)

#Clearing memory
gc.collect()

#Using the model on the testing set
rf_au_50_y_pred = rf_au_50.predict(x_au_test)

#Checking the model accuracy
rf_au_50_acc = metrics.accuracy_score(y_au_test, rf_au_50_y_pred)
print(rf_au_50_acc)

#Clearing memory
gc.collect()

#Backing up data

#Loading custom pickle functions
#The relative path 'Scripts/Pickle Backup.py' will only work if you are in
#the correct working directory which contains the subfoler 'Scripts' and
#the file "Pickle Backup.py" within that subfolder
exec(open('Scripts/Pickle Backup.py').read())

#Executing function defined for creating/updating .pkl backup, defined in
#Pickle Backup.py

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
           hivedata_spec_long,
           svm_1_lin_acc,
           svm_1_rbf_acc,
           svm_1_poly_acc,
           svm_1_sig_acc,
           au_feat_col,
           hivedata_au_feat,
           rf_au_20_acc,
           rf_au_50_acc,
           rf_wea_20_acc,
           rf_wea_50_acc,
           rf_wea_20_y_pred,
           rf_wea_50_y_pred,
           rf_au_20_y_pred,
           rf_au_50_y_pred)

#Model Visualisation

#Weather Based Model

#Since the model 'rf_wea_50' had a better accuracy, it shall be used
#for the visualisations

#Visualising model accuracy segregated by Queen Status classes

# Calculate the percentage of true and false predictions for each level
levels = ['Original Queen',
          'Absent',
          'Present & Rejected',
          'Present & Newly Accepted']
true_percentages = []
false_percentages = []

for level in range(4):
    true_count = np.sum((y_wea_test == level) & (rf_wea_50_y_pred == level))
    false_count = np.sum((y_wea_test == level) & (rf_wea_50_y_pred != level))
    total_count = np.sum(y_wea_test == level)
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100
    true_percentages.append(true_percentage)
    false_percentages.append(false_percentage)

# Create the bar plot
x = np.arange(len(levels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, false_percentages, width, color='red', label='False')
rects2 = ax.bar(x + width/2, true_percentages, width, color='green', label='True')

# Set axis labels and title
ax.set_xlabel('Queen Status')
ax.set_ylabel('Percentage')
ax.set_title('RF Prediction Accuracy (Weather Based Model)')
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.legend()

# Add percentage labels above each bar
for rect1, rect2 in zip(rects1, rects2):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    ax.annotate(f'{height1:.1f}%', 
                xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                xytext=(0, 4), 
                textcoords="offset points", 
                ha='center', 
                va='bottom')
    ax.annotate(f'{height2:.1f}%', 
                xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                xytext=(0, 4), 
                textcoords="offset points", 
                ha='center', 
                va='bottom')

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/rf_wea_50_acc.png")

plt.show()


#Visualising feature importance in the random forest model

#Storing feature importance
feat_rf_wea_50 = rf_wea_50.feature_importances_

#Creating filter to sort feature importance in descending order
imp = np.argsort(feat_rf_wea_50)[::-1]

# Create a bar plot of the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feat_rf_wea_50)), 
        feat_rf_wea_50[imp])
plt.xticks(range(len(feat_rf_wea_50)), 
           [weather_columns[:-1][i] for i in imp], 
           rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance by RF (Weather Based Model)')
plt.tight_layout()

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/rf_wea_50_feat.png")

plt.show()

#Audio Based Models

#Since the model 'rf_au_50' had a better accuracy, it shall be used for the
#visualisations

#Visualising model accuracy segregated by Queen Status classes

# Calculate the percentage of true and false predictions for each level
levels = ['Original Queen',
          'Absent',
          'Present & Rejected',
          'Present & Newly Accepted']
true_percentages = []
false_percentages = []

for level in range(4):
    true_count = np.sum((y_au_test == level) & (rf_au_50_y_pred == level))
    false_count = np.sum((y_au_test == level) & (rf_au_50_y_pred != level))
    total_count = np.sum(y_au_test == level)
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100
    true_percentages.append(true_percentage)
    false_percentages.append(false_percentage)

# Create the bar plot
x = np.arange(len(levels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, false_percentages, width, color='red', label='False')
rects2 = ax.bar(x + width/2, true_percentages, width, color='green', label='True')

# Set axis labels and title
ax.set_xlabel('Queen Status')
ax.set_ylabel('Percentage')
ax.set_title('RF Prediction Accuracy (Audio Feature Based Model)')
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.legend()

# Add percentage labels above each bar
for rect1, rect2 in zip(rects1, rects2):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    
    ax.annotate(f'{height1:.1f}%', 
                xy=(rect1.get_x() + rect1.get_width() / 2, height1),
                xytext=(0, 4), 
                textcoords="offset points", 
                ha='center', 
                va='bottom')
    
    ax.annotate(f'{height2:.1f}%', 
                xy=(rect2.get_x() + rect2.get_width() / 2, height2),
                xytext=(0, 4), 
                textcoords="offset points", 
                ha='center', 
                va='bottom')

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/rf_au_50.png")

plt.show()

#Visualising feature importance in the random forest model

#Storing feature importance
feat_rf_au_50 = rf_au_50.feature_importances_

#Creating filter to sort feature importance in descending order
imp = np.argsort(feat_rf_au_50)[::-1]

# Create a bar plot of the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feat_rf_au_50)),
        feat_rf_au_50[imp])
plt.xticks(range(len(feat_rf_au_50)),
           [au_feat_col[:-1][i] for i in imp], 
           rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance by RF (Audio Feature Based Model)')
plt.tight_layout()

#Saving plot
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/rf_au_50_feat.png")

plt.show()


#Model Diagnostics

#Weather Based models

#20 Estimators

# Confusion Matrix
conf_matrix_rf_20 = confusion_matrix(y_wea_test, rf_wea_20_y_pred)
print("Confusion Matrix (RF max_samples=0.2):")
print(conf_matrix_rf_20)

# Classification Report
class_report_rf_20 = classification_report(y_wea_test, rf_wea_20_y_pred)
print("Classification Report (RF max_samples=0.2):")
print(class_report_rf_20)

# Feature Importance
feature_importance_rf_20 = rf_wea_20.feature_importances_
print("Feature Importance (RF max_samples=0.2):")
print(feature_importance_rf_20)

#50 Estimators

# Confusion Matrix
conf_matrix_rf_50 = confusion_matrix(y_wea_test, rf_wea_50_y_pred)
print("Confusion Matrix (RF max_samples=0.2):")
print(conf_matrix_rf_50)

# Classification Report
class_report_rf_50 = classification_report(y_wea_test, rf_wea_50_y_pred)
print("Classification Report (RF max_samples=0.2):")
print(class_report_rf_50)

# Feature Importance
feature_importance_rf_50 = rf_wea_20.feature_importances_
print("Feature Importance (RF max_samples=0.2):")
print(feature_importance_rf_50)


#Audio Feature Based Models

#20 Estimators

# Confusion Matrix
conf_matrix_rf_20 = confusion_matrix(y_au_test, rf_au_20_y_pred)
print("Confusion Matrix (RF max_samples=0.2):")
print(conf_matrix_rf_20)

# Classification Report
class_report_rf_20 = classification_report(y_au_test, rf_au_20_y_pred)
print("Classification Report (RF max_samples=0.2):")
print(class_report_rf_20)

# Feature Importance
feature_importance_rf_20 = rf_au_20.feature_importances_
print("Feature Importance (RF max_samples=0.2):")
print(feature_importance_rf_20)

#50 Estimators

# Confusion Matrix
conf_matrix_rf_50 = confusion_matrix(y_au_test, rf_au_50_y_pred)
print("Confusion Matrix (RF max_samples=0.2):")
print(conf_matrix_rf_50)

# Classification Report
class_report_rf_50 = classification_report(y_au_test, rf_au_50_y_pred)
print("Classification Report (RF max_samples=0.2):")
print(class_report_rf_20)

# Feature Importance
feature_importance_rf_50 = rf_au_50.feature_importances_
print("Feature Importance (RF max_samples=0.2):")
print(feature_importance_rf_50)

#Combining classification report for audio feature based models

# Assuming you have the classification reports already computed
report_rf_au_20 = classification_report(y_au_test, rf_au_20_y_pred, output_dict=True)
report_rf_au_50 = classification_report(y_au_test, rf_au_50_y_pred, output_dict=True)
report_rf_wea_20 = classification_report(y_wea_test, rf_wea_20_y_pred, output_dict=True)
report_rf_wea_50 = classification_report(y_wea_test, rf_wea_50_y_pred, output_dict=True)

# Extract metrics of class '2' from each classification report
metrics_class_2_rf_au_20 = report_rf_au_20['2']
metrics_class_2_rf_au_50 = report_rf_au_50['2']
metrics_class_2_rf_wea_20 = report_rf_wea_20['2']
metrics_class_2_rf_wea_50 = report_rf_wea_50['2']

# Create a DataFrame with the metrics
rf_metrics = pd.DataFrame({'Metric': metrics_class_2_rf_au_20.keys(), 'Audio (RF_20)': [round(value, 2) for value in metrics_class_2_rf_au_20.values()], 'Audio (RF_50)': [round(value, 2) for value in metrics_class_2_rf_au_50.values()], 'Weather (RF_20)': [round(value, 2) for value in metrics_class_2_rf_wea_20.values()], 'Weather (RF_50)': [round(value, 2) for value in metrics_class_2_rf_wea_50.values()]})

# Display the DataFrame
#cap = "Statistical Comparison of Random Forest Models"
#rf_table = rf_metrics.iloc[:-1,:].style.set_caption(cap)
#display(rf_table)
print(rf_metrics.iloc[:-1,:])

#Extracting feature importance from model (for best model)
rf_50_feat_imp = rf_wea_pipe_50.named_steps['randomforestclassifier'].feature_importances_

#Storing feature importance in a dataset and printing it as a table
rf_feat_imp = pd.DataFrame({'Predictor': weather_columns[:-1],'Feature Importance': [round(value, 2) for value in rf_50_feat_imp]})

print(rf_feat_imp)


#Saving data
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
           svm_metrics,
           au_feat_col,
           hivedata_au_feat,
           rf_au_20_acc,
           rf_au_50_acc,
           rf_wea_20_acc,
           rf_wea_50_acc,
           rf_wea_20_y_pred,
           rf_wea_50_y_pred,
           rf_au_20_y_pred,
           rf_au_50_y_pred,
           rf_metrics,
           rf_feat_imp)