with open('Backups/beehive_old.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)
date_old = loaded_objects['hivedata']['date']

#Splitting date and time
hivedata[['date', 'time']] = hivedata['date'].str.split(' ', n = 1, expand=True)

#Correcting format of date column and time column

#Format %Y-%m-%d
hivedata['date'] = pd.to_datetime(hivedata['date'])
#Removing extra 'time' component from date values
hivedata['date'] = hivedata['date'].dt.strftime('%Y-%m-%d')

#Format %H:%M%:%S
hivedata['time'] = pd.to_datetime(hivedata['time'], format = '%H:%M:%S')
#Removing extra 'date' component from time values
hivedata['time'] = hivedata['time'].dt.time

#Updating labels for queen related columns

#Queen Acceptance
def queen_accept(value):
    if value == "Queen Absent":
        return 0
    elif value == "Queen Rejected":
        return 1
    else:
        return 2
#Applying function for queen acceptance
hivedata["queen acceptance"] = hivedata["queen acceptance"].apply(queen_accept)

#Queen Presence
def queen_pres(value):
    if value == "Queen Absent":
        return 0
    else:
        return 1
    
hivedata["queen presence"] = hivedata["queen presence"].apply(queen_pres)

#Queen Status
def queen_stat(value):
    if value == "Original Queen":
        return 0
    elif value == "Queen Absent":
        return 1
    elif value == "Present & Rejected":
        return 2
    else:
        return 3

hivedata["queen status"] = hivedata["queen status"].apply(queen_stat)

#Trying out different kernels for models based on audio features

# Defining relevant audio features as predictors
x = hivedata_au_feat.iloc[:,:-1]
y = hivedata_au_feat.iloc[:,-1]

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=110)

#Defining weights for imbalanced dataset
svm_weights = compute_class_weight(class_weight = 'balanced',
                                   classes = np.unique(y_train), 
                                   y = y_train)
svm_weights = dict(enumerate(svm_weights))

# Define the kernel types
kernels = ['linear', 'rbf', 'poly', 'sigmoid']



# Initialize dictionaries to store accuracy scores for each class and model
all_acc_scores = {k: {} for k in kernels}

# Iterate through classes and models
for queen_label in range(4):
    for k in kernels:
        # Training SVM Model
        svm = SVC(kernel= k, class_weight=svm_weights)
        svm.fit(x_train, y_train)
        
        # Predicting based on the test set
        y_pred = svm.predict(x_test)
        
        # Computing model accuracy
        acc = metrics.accuracy_score(y_test[y_test == queen_label], 
                                     y_pred[y_test == queen_label])
        
        # Store accuracy in the dictionary
        all_acc_scores[k][queen_label] = acc
        
        # Clearing memory
        gc.collect()

# Create a DataFrame from the accuracy_scores dictionary
svm_au_all_acc = pd.DataFrame(all_acc_scores)

# Display the DataFrame
print(svm_au_all_acc)


#PCA Based Cluster Visualisation

#Model 1- Audio Features

#Reducing dimensions using PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_test)

# Colour map for 4 classes of queen status
colours = ['red', 'green', 'blue', 'yellow']

#Plot
for class_label in range(4):
    plt.scatter(x_pca[y_test == class_label, 0],
                x_pca[y_test == class_label, 1],
                c=colours[class_label],
                label=f'Class {class_label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title("Clustering from SVM Model Using PCA (Audio Features)")
plt.legend()
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/svm.png")
plt.show()


#Creating plot to visualise SVM performance

# Calculate the percentage of true and false predictions for each level
levels = ['Original Queen',
          'Absent',
          'Present & Rejected',
          'Present & Newly Accepted']

true_percentages = []
false_percentages = []

for level in range(4):
    true_count = np.sum((y_test == level) & (y_pred == level))
    false_count = np.sum((y_test == level) & (y_pred != level))
    total_count = np.sum(y_test == level)
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100
    true_percentages.append(true_percentage)
    false_percentages.append(false_percentage)

# Create the bar plot
x = np.arange(len(levels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, 
                false_percentages,
                width, color='red', 
                label='False')
rects2 = ax.bar(x + width/2, 
                true_percentages, 
                width, color='green', 
                label='True')

# Set axis labels and title
ax.set_xlabel('Status')
ax.set_ylabel('Percentage')
ax.set_title('SVM Prediction Accuracy (Spectrogram Based Model)')
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.legend()

# Add percentage labels above each bar
for rect1, rect2 in zip(rects1, rects2):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    
    ax.annotate(f'{height1:.1f}%', 
                xy=(rect1.get_x() + rect1.get_width() / 2, 
                    height1),
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
plt.savefig("D:/Python/Spyder/Honey Bee/Visualisations/svm_spec.png")

plt.show()

# Create a scatter plot
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap=plt.cm.Paired)

# Add labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification Results")

# Show the color legend
plt.colorbar()

# Display the plot
plt.show()


#Defining image locations for plots
svm_au = mpimg.imread("D:/Python/Spyder/Honey Bee/Visualisations/svm.png")
svm_spec = mpimg.imread("D:/Python/Spyder/Honey Bee/Visualisations/svm_spec.png")

plt.figure(figsize = (12, 4))

#Part 1- SVM Clustering- Audio Features
plt.subplot(1, 2, 1)
plt.imshow(svm_au)
plt.axis('off')
plt.title('SVM based on Audio Features')

#Part 2- SVM Clustering- Spectrograms
plt.subplot(1, 2, 2)
plt.imshow(svm_spec)
plt.axis('off')
plt.title('SVM based on Spectrograms')

#Part 3- SVM Clustering- Weather
plt.subplot(1, 2, 2)
plt.imshow(svm_spec)
plt.axis('off')
plt.title('SVM based on Spectrograms')

#Part 4- Blank Space for 2X2 grid

plt.tight_layout()

#Displaying the combined figure
plt.show()

#Weights have been calculated in 'Support Vector Machine.py' which is to be
#executed before 'Gradient Boosting.py'. The pickle backup contains the weights
#dictionary


#Creating summary of model diagnostics with class '2'- Queen Present & Rejected
#Combining Classification Reports

#Defining Training and Testing Datasets (Set Seed: 110)

#For model 1- Audio Features

#Defining predictors and target variable
x = hivedata_au_feat.iloc[:,:-1]
y = hivedata_au_feat.iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_1 = train_test_split(x, y, test_size=0.2, random_state=110)


#For model 3- Weather Data

#Defining predictors and target variable
x = hivedata_weather.fillna(method = 'ffill').iloc[:,:-1]
y = hivedata_weather.fillna(method = 'ffill').iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_3 = train_test_split(x, y, test_size=0.2, random_state=110)


#For model 4- Audio features and Weather Data

#Defining predictors and target variable
x = hivedata_wea_spec.fillna(method = 'ffill').iloc[:,:-1]
y = hivedata_wea_spec.fillna(method = 'ffill').iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_4 = train_test_split(x, y, test_size=0.2, random_state=110)


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



######################################################


svm_1_all_acc = pd.DataFrame({'Kernel':['Linear', 'RBF', 'Polynomial', 'Sigmoid'], 'Accuracy':[svm_1_lin_acc, svm_1_rbf_acc, svm_1_poly_acc, svm_1_sig_acc]})

#UNWANTED CHUNKS

#1

{python Exploration_1, echo=FALSE, message=FALSE, warning=FALSE}

(
ggplot(data = hivedata, mapping = aes(x = 'device')) +
  geom_bar()+
  scale_x_continuous(breaks=range(1, 3))+
  ggtitle("Split of Rows Recorded by Devices") +
  xlab("Device Number") +
  ylab("Record Count")+
  theme(plot_title = element_text(hjust = 0.5, face = "bold"))
)

#2

{python Exploration_2, echo=FALSE, message=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'hive number')) + 
  geom_bar()
)

#3

{python Exploration_3, echo=FALSE, warning=FALSE}

#Note that the following will only work after the file 'Data Wrangling.py'
#has been executed, as the date column required some basic formatting.

#Extracting month from date-time column
month = hivedata['date'].dt.month

#Visualising months covered in the dataset

(
  ggplot(data=hivedata, mapping = aes(x = month)) + 
  geom_bar() +
  scale_x_continuous(breaks=range(1, 13))
)

#4

{python Exploration_4, echo=FALSE, warning=FALSE, exec=FALSE}
#EXECUTE IS FALSE IN THIS CHUNK
(
  ggplot(data = hivedata, mapping = aes(x = 'hive number', y = 'hive temp'))+
  geom_density()+
  facet_wrap("~ 'hive number'")
)

#5

{python Exploration_5, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'hive humidity')) + 
  geom_density()
)

#6

{python Exploration_6, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'hive pressure')) + 
  geom_density()
)

#7

{python Exploration_7, echo=FALSE, warning=FALSE}

#month_day = hivedata['date'].dt.day
(
  ggplot(data = hivedata, mapping = aes(x = 'weather temp')) + 
  geom_density()
)

#8

{python Exploration_8, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'wind speed')) + 
  geom_density()
)

#9

{python Exploration_9, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'gust speed')) + 
  geom_density()
)

#10

{python Exploration_10, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'cloud coverage')) + 
  geom_density()
)

#11

{python Exploration_13, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'frames')) + 
  geom_bar()
)

#Segment based on  hive number

#12

{python Exploration_14, echo=FALSE, warning=FALSE}

(
  ggplot(data = hivedata, mapping = aes(x = 'target')) + 
  geom_bar() + 
  scale_x_continuous(breaks=range(0, 6))
)

#Add labels

#13

{python, echo=FALSE, warning=FALSE}
#Check for ways to visualise time
#ggplot(data = hivedata, mapping = aes(x = 'target')) + geom_bar()

#14

#Defining image locations for plots
UMAP_2D = mpimg.imread("D:/Python/Spyder/Honey Bee/Visualisations/2d_umap.png")
UMAP_3D = mpimg.imread("D:/Python/Spyder/Honey Bee/Visualisations/3d_umap.png")

plt.figure(figsize = (12, 4))

#Part 1- 2D UMAP 
plt.subplot(1, 2, 1)
plt.imshow(UMAP_2D)
plt.axis('off')
plt.title('2D UMAP')

#Part 2- 3D UMAP 
plt.subplot(1, 2, 2)
plt.imshow(UMAP_3D)
plt.axis('off')
plt.title('3D UMAP')

plt.tight_layout()

# Show the combined figure
plt.show()

#15

#Defining image locations for plots
svm_au = mpimg.imread("D:/Python/Spyder/Honey Bee/Visualisations/svm.png")
svm_spec = mpimg.imread("D:/Python/Spyder/Honey Bee/Visualisations/svm_spec.png")

plt.figure(figsize = (12, 4))

#Part 1- 2D UMAP 
plt.subplot(1, 2, 1)
plt.imshow(svm_au)
plt.axis('off')
plt.title('SVM based on Audio Features')

#Part 2- 3D UMAP 
plt.subplot(1, 2, 2)
plt.imshow(svm_spec)
plt.axis('off')
plt.title('SVM based on Spectrograms')

plt.tight_layout()

#Displaying the combined figure
plt.show()

##############################################################################
##############################################################################
#EXTRA CODE REMOVED FROM REPORT
#MOSTLY TABLES

###########################################################################
svm_1_all_acc = pd.DataFrame({'Kernel':['Linear', 'RBF', 'Polynomial', 'Sigmoid'], 'Accuracy':[svm_1_lin_acc, svm_1_rbf_acc, svm_1_poly_acc, svm_1_sig_acc]})


###########################################################################
#Creating summary of model diagnostics with class '2'- Queen Present & Rejected
#Combining Classification Reports

#Defining Training and Testing Datasets (Set Seed: 110)

#For model 1- Audio Features

#Defining predictors and target variable
x = hivedata_au_feat.iloc[:,:-1]
y = hivedata_au_feat.iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_1 = train_test_split(x, y, test_size=0.2, random_state=110)


#For model 3- Weather Data

#Defining predictors and target variable
x = hivedata_weather.fillna(method = 'ffill').iloc[:,:-1]
y = hivedata_weather.fillna(method = 'ffill').iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_3 = train_test_split(x, y, test_size=0.2, random_state=110)


#For model 4- Audio features and Weather Data

#Defining predictors and target variable
x = hivedata_wea_spec.fillna(method = 'ffill').iloc[:,:-1]
y = hivedata_wea_spec.fillna(method = 'ffill').iloc[:,-1]

#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test_4 = train_test_split(x, y, test_size=0.2, random_state=110)


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

###############################################################################

#Defining training and testing sets for weather and audio feature based models
#Set Seed: 110

#Weather
x_wea = hivedata_weather.iloc[:, :-1].values
y_wea = hivedata_weather.iloc[:, -1].values

x_wea_train, x_wea_test, y_wea_train, y_wea_test = train_test_split(x_wea, y_wea, test_size=0.2, train_size=0.8, random_state=110)

#Audio Features
x_au = hivedata_au_feat.iloc[:, :-1].values
y_au = hivedata_au_feat.iloc[:, -1].values

x_au_train, x_au_test, y_au_train, y_au_test = train_test_split(x_au, y_au, test_size=0.2, train_size=0.8, random_state=110)

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

# Create a DataFrame with the diagnostic metrics
rf_metrics = pd.DataFrame({'Metric': metrics_class_2_rf_au_20.keys(), 'Audio (RF_20)': [round(value, 2) for value in metrics_class_2_rf_au_20.values()], 'Audio (RF_50)': [round(value, 2) for value in metrics_class_2_rf_au_50.values()], 'Weather (RF_20)': [round(value, 2) for value in metrics_class_2_rf_wea_20.values()], 'Weather (RF_50)': [round(value, 2) for value in metrics_class_2_rf_wea_50.values()]})

# Display the DataFrame
#cap = "Statistical Comparison of Random Forest Models"
#rf_table = rf_metrics.iloc[:-1,:].style.set_caption(cap)
#display(rf_table)

###############################################################################

#Defining imputer to impute missing values in the numeric columns using medians
imp_median = SimpleImputer(strategy = 'median')

#Attempting to use a heavier model on the weather data
rf_wea_50 = RandomForestClassifier(max_depth= 10, 
                               max_samples= 0.5,
                               max_features= 5,
                               n_estimators = 50,
                               warm_start=True)

#Building pipeline for imputer to work with rf_wea_50
rf_wea_pipe_50 = make_pipeline(imp_median, rf_wea_50)

#Fitting the model
rf_wea_pipe_50.fit(x_wea_train, y_wea_train)

#Using the model on the testing set
rf_wea_50_y_pred = rf_wea_pipe_50.predict(x_wea_test)

#Extracting feature importance from model (for best model)
rf_50_feat_imp = rf_wea_pipe_50.named_steps['randomforestclassifier'].feature_importances_

#Storing feature importance in a dataset and printing it as a table
feat_imp = pd.DataFrame({'Predictor': weather_columns[:-1],'Feature Importance': [round(value, 2) for value in rf_50_feat_imp]})

##############################################################################


#Creating training and testing sets
x = hivedata_wea_spec.iloc[:, :-1]
y = hivedata_wea_spec.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 110)

#Classification report
xgb_report = classification_report(y_test, xgb_y_pred_1)

##############################################################################

#Defining training and testing sets for weather and audio feature based models

#Audio features and weather
x = hivedata_wea_spec.iloc[:, :-1]
y = hivedata_wea_spec.iloc[:, -1]

x_train, x_test, y_train, y_test_1 = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 110)

#Audio features                                                      
x = hivedata_au_feat.iloc[:, :-1]
y = hivedata_au_feat.iloc[:, -1]

x_train, x_test, y_train, y_test_2 = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 110)                                                      

#Weather
x = hivedata_weather.iloc[:, :-1]
y = hivedata_weather.iloc[:, -1]

x_train, x_test, y_train, y_test_3 = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 110) 

#Combining classification report for audio feature based models

# Assuming you have the classification reports already computed
xgb_report_1 = classification_report(y_test_1, xgb_y_pred_1, output_dict=True)
xgb_report_2 = classification_report(y_test_2, xgb_y_pred_2, output_dict=True)
xgb_report_3 = classification_report(y_test_3, xgb_y_pred_3, output_dict=True)

# Extract metrics of class '2' from each classification report
metrics_class_2_xgb_1 = xgb_report_1['2']
metrics_class_2_xgb_2 = xgb_report_2['2']
metrics_class_2_xgb_3 = xgb_report_3['2']

# Create a DataFrame with the diagnostic metrics
xgb_metrics = pd.DataFrame({'Metric': metrics_class_2_xgb_1.keys(), 'Audio & Weather': [round(value, 2) for value in metrics_class_2_xgb_1.values()], 'Audio': [round(value, 2) for value in metrics_class_2_xgb_2.values()], 'Weather': [round(value, 2) for value in metrics_class_2_xgb_3.values()]})

#############################################################################

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

#Fitting Gradient Boosting model
xgb_3.fit(x_train, y_train)

#Extracting feature importance from model (for best model)
xgb_3_feat_imp = xgb_3.feature_importances_

#Storing feature importance in a dataset and printing it as a table
xgb_3_feat_imp = pd.DataFrame({'Predictor': weather_columns[:-1],'Feature Importance': [round(value, 2) for value in xgb_3_feat_imp]})


#Fe

#Processing Audio features for melted dataset
#A function has been previosly defined for extracting audio features

for index, row in hivedata_spec_long.iterrows():
    # Extract audio features
    audio_file = row['audio']
    
    #Using previously defined function here
    audio_features = extract_audio_features(audio_file)
    
    # Update the DataFrame with the audio features
    for feature, feature_values in audio_features.items():
        column_name = f'{feature}'
        hivedata_spec_long.loc[index, column_name] = np.mean(feature_values)
        
        

#BAYESIAN MODEL

tfd = tfp.distributions

# Split the dataset into train and test sets
train_data, test_data = train_test_split(hivedata_spec_long, test_size=0.2, random_state=110)

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
train_data['queen_status'] = train_data['queen_status']
test_data['queen_status'] = test_data['queen_status']
train_labels = train_data['queen_status']
test_labels = test_data['queen_status']

# Calculate class weights
weights_3 = compute_class_weight(class_weight ='balanced', 
                               classes = np.unique(train_labels), 
                               y = train_labels)
weights_3 = dict(enumerate(weights_3))

# Define the data generators for training and testing
batch_size = 64
train_generator = image_gen.flow_from_dataframe(
    train_data,
    x_col='spec_loc',
    y_col='queen_status',
    batch_size=batch_size,
    class_mode='raw',
    target_size=(224, 224)
)

test_generator_3 = image_gen.flow_from_dataframe(
    test_data,
    x_col='spec_loc',
    y_col='queen_status',
    batch_size=batch_size,
    class_mode='raw',
    target_size=(224, 224)
)

# Build the Bayesian CNN model
model_3 = Sequential([
    # Convolutional layer 1
    tfp.layers.Convolution2DFlipout(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    # Convolutional layer 2
    tfp.layers.Convolution2DFlipout(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    # Convolutional layer 3
    tfp.layers.Convolution2DFlipout(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    # Flattening images
    Flatten(),

    # Bayesian Dense layer 1
    tfp.layers.DenseFlipout(256, activation='relu'),

    # Output with DistributionLambda
    tfp.layers.DistributionLambda(lambda t: tfd.Categorical(logits=t))
])

# Define the negative log likelihood loss function
def negative_log_likelihood(y_true, y_pred, class_weights):
    y_true_np = tf.keras.backend.get_value(y_true)  # Convert tensor to numpy array
    negative_log_prob = -y_pred.log_prob(y_true)
    weighted_loss = tf.reduce_mean(negative_log_prob * class_weights[y_true_np])
    return weighted_loss

# Compile the model with the new loss function
model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                loss=lambda y_true, y_pred: negative_log_likelihood(y_true, y_pred, weights_3), 
                metrics=['accuracy'])

# Train the model
cnn_history_3 = model_3.fit(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=15,
    validation_data=test_generator_3,
    validation_steps=len(test_data) // batch_size,
    class_weight=weights_3
)

# Evaluate the model on the test set
test_loss_cnn_3, cnn_acc_3 = model_3.evaluate(test_generator_3)
print(f"Test accuracy: {cnn_acc_3 * 100:.2f}%")

# Garbage collection (clearing RAM)
gc.collect()

pros = ['-',
  '-',
  '-', 
  '-', 
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',]

cons = ['Unclear separation of clusters',
  'Unclear separation of clusters',
  '-', 
  '-', 
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',
  '-',]


#Centre alignment
display_width = pd.get_option('display.width')
title_padding = (display_width - len(title)) // 2
table_padding = 

print(model_sum.to_markdown(index = False))

model_sum_tab = tabulate(model_sum, tablefmt="fancy_outline", headers='keys')

#Adding title
title = "Comparison of Models"

#Manually centre aligning table based on page width
#Centre alignment
page_width = pd.get_option('display.width')
title_padding = (page_width - len(title)) // 2
table_padding = (page_width - len(model_sum_tab.split('\n')[0])) // 2

#Printing table
print(' ' * title_padding + title)
print(model_sum.to_markdown(index = False))


{python UMAP_Plots, echo=FALSE, message=FALSE, warning=FALSE}

from plotnine import *
from plotnine import ggplot, geom_raster, labs, theme_void, theme, facet_grid
from PIL import Image

# Defining image locations for plots
UMAP_2D_path = "D:/Python/Spyder/Honey Bee/Visualisations/2d_umap.png"
UMAP_3D_path = "D:/Python/Spyder/Honey Bee/Visualisations/3d_umap.png"

# Read images using PIL
umap_2d_image = Image.open(UMAP_2D_path)
umap_3d_image = Image.open(UMAP_3D_path)

# Create separate ggplots for each image
umap_2d_plot = (
    ggplot() +
    geom_raster(image=umap_2d_image) +
    labs(title="2D UMAP") +
    theme_void()
)

umap_3d_plot = (
    ggplot() +
    geom_raster(image=umap_3d_image) +
    labs(title="3D UMAP") +
    theme_void()
)

# Create a grid layout using facet_grid
(
    ggplot() +
    theme(figure_size=(12, 4)) +
    facet_grid('.~.') +  # Create a grid layout
    geom_raster(data={'image': [umap_2d_image, umap_3d_image]}, aes(image='image')) +
    labs(title=None) +
    theme_void()
)

{r XGB_Model_Diagnostics_1, echo=FALSE, message=FALSE, warning=FALSE}

#Displaying Classification Report for Audio Features & Weather based model

kable(py$as.dataframe(xgb_report))


{r, warning=FALSE, message=FALSE, echo=FALSE}

#Manually typed table because of an unknown error while rendering. 
#Values taken manually from actual diagnostic table
kable(py$xgb_3_feat_imp)