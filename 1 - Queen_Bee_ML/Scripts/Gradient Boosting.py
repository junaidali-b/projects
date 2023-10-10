#Importing required libraries for modelling gradient boosting models and 
#visualising results

import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
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

#Model 1: Audio features and weather columns used

#Subset of hivedata dataset where the the weather related columns and
#spectrogram columns have been added.

hivedata_wea_spec = hivedata[weather_columns[:-1] + au_feat_col]

#Creating training and testing sets
x = hivedata_wea_spec.iloc[:, :-1]
y = hivedata_wea_spec.iloc[:, -1]

x_train, x_test, y_train, y_test_1 = train_test_split(x,
                                                      y,
                                                      test_size = 0.2,
                                                      train_size = 0.8,
                                                      random_state = 110)

#Defining Gradient Boosting model

#Defining sample weights (Each sample gets its own weight instead of one 
#weight per class)

queen_status_weights = class_weight.compute_sample_weight(class_weight = 'balanced',
                                                          y = y_train)

xgb_1 = xgb.XGBClassifier()

#Clearing RAM
gc.collect()

#Fitting Gradient Boosting model
xgb_1.fit(x_train, y_train, sample_weight = queen_status_weights)

#Using model on test set to make predictions
xgb_y_pred_1 = xgb_1.predict(x_test)

#Checking model accuracy
xgb_acc_1 = accuracy_score(y_test_1, xgb_y_pred_1)
print(xgb_acc_1)


#Model 2: Audio features used

#Creating training and testing sets
x = hivedata_au_feat.iloc[:, :-1]
y = hivedata_au_feat.iloc[:, -1]

x_train, x_test, y_train, y_test_2 = train_test_split(x,
                                                      y,
                                                      test_size = 0.2,
                                                      train_size = 0.8,
                                                      random_state = 110)

#Defining Gradient Boosting model
xgb_2 = xgb.XGBClassifier()

#Clearing RAM
gc.collect()

#Fitting Gradient Boosting model
#Weights are common for all models because all subsets have the same number of
#rows the same array for the target variable, and the same seed for spilitting
#training and testing datasets

xgb_2.fit(x_train, y_train, sample_weight = queen_status_weights)

#Using model on test set to make predictions
xgb_y_pred_2 = xgb_2.predict(x_test)

#Checking model accuracy
xgb_acc_2 = accuracy_score(y_test_2, xgb_y_pred_2)
print(xgb_acc_2)


#Model 3: Weather columns used

#Creating training and testing sets
x = hivedata_weather.iloc[:, :-1]
y = hivedata_weather.iloc[:, -1]

x_train, x_test, y_train, y_test_3 = train_test_split(x,
                                                      y,
                                                      test_size = 0.2,
                                                      train_size = 0.8,
                                                      random_state = 110)

#Defining Gradient Boosting model
xgb_3 = xgb.XGBClassifier()

#Clearing RAM
gc.collect()

#Fitting Gradient Boosting model
#Weights are common for all models because all subsets have the same number of
#rows the same array for the target variable, and the same seed for spilitting
#training and testing datasets

xgb_3.fit(x_train, y_train, sample_weight = queen_status_weights)

#Using model on test set to make predictions
xgb_y_pred_3 = xgb_3.predict(x_test)

#Checking model accuracy
xgb_acc_3 = accuracy_score(y_test_3, xgb_y_pred_3)
print(xgb_acc_3)


#Model Diagnostics

#Model 1

#Confusion matrix
cm_1 = confusion_matrix(y_test_1, xgb_y_pred_1)
sns.heatmap(cm_1, annot=True, fmt='d', cmap='Greens')

#Classification report
xgb_report_1 = classification_report(y_test_1, xgb_y_pred_1)
print(xgb_report_1)


#Model 2

#Confusion matrix
cm_2 = confusion_matrix(y_test_2, xgb_y_pred_2)
sns.heatmap(cm_2, annot=True, fmt='d', cmap='Greens')

#Classification report
xgb_report_2 = classification_report(y_test_2, xgb_y_pred_2)
print(xgb_report_2)


#Model 1

#Confusion matrix
cm_3 = confusion_matrix(y_test_3, xgb_y_pred_3)
sns.heatmap(cm_3, annot=True, fmt='d', cmap='Greens')

#Classification report
xgb_report_3 = classification_report(y_test_3, xgb_y_pred_3)
print(xgb_report_3)

#Combining Classification Reports

# Assuming you have the classification reports already computed
xgb_report_1 = classification_report(y_test_1, xgb_y_pred_1, output_dict=True)
xgb_report_2 = classification_report(y_test_2, xgb_y_pred_2, output_dict=True)
xgb_report_3 = classification_report(y_test_3, xgb_y_pred_3, output_dict=True)

# Extract metrics of class '2' from each classification report
metrics_class_2_xgb_1 = xgb_report_1['2']
metrics_class_2_xgb_2 = xgb_report_2['2']
metrics_class_2_xgb_3 = xgb_report_3['2']

# Create a DataFrame with the diagnostic metrics
xgb_metrics = pd.DataFrame({'Metric': metrics_class_2_xgb_1.keys(), 
                            'Audio & Weather': [round(value, 2) for value in metrics_class_2_xgb_1.values()], 
                            'Audio': [round(value, 2) for value in metrics_class_2_xgb_2.values()], 
                            'Weather': [round(value, 2) for value in metrics_class_2_xgb_3.values()]})

print(xgb_metrics.iloc[:-1,:])

#Classification report for first model

xgb_report = classification_report(y_test_1, xgb_y_pred_1)

#Feature importance for weather based model
#Extracting feature importance from model (for best model)
xgb_3_feat_imp = xgb_3.feature_importances_

#Storing feature importance in a dataset and printing it as a table
xgb_3_feat_imp = pd.DataFrame({'Predictor': weather_columns[:-1],
                               'Feature Importance': [round(value, 2) for value in xgb_3_feat_imp]})

#Updating backup data with gradient boosting dataset and accuracy.
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
           svm_1_all_acc,
           svm_metrics,
           rf_au_20_acc,
           rf_au_50_acc,
           rf_wea_20_acc,
           rf_wea_50_acc,
           rf_wea_20_y_pred,
           rf_wea_50_y_pred,
           rf_au_20_y_pred,
           rf_au_50_y_pred,
           rf_metrics,
           rf_feat_imp,
           hivedata_wea_spec,
           xgb_acc_1,
           xgb_y_pred_1,
           xgb_acc_2,
           xgb_y_pred_2,
           xgb_acc_3,
           xgb_y_pred_3,
           xgb_report,
           xgb_metrics,
           xgb_3_feat_imp)