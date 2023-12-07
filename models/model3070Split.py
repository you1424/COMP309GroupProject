# =============================================================================
# LOAD DATA
# =============================================================================
import pandas as pd
import os
path = "C:/Users/ivanz/Downloads/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
data_group8 = pd.read_csv(fullpath)

# Remove UNKNOWN's from Status column
data_group8 = data_group8.loc[data_group8["Status"] != 'UNKNOWN']
# Reference Status column to 0 or 1
data_group8['Status'].replace(['STOLEN', 'RECOVERED'],[0, 1], inplace=True)

# keep columns that we will be use for our data modeling
data_group8 = data_group8[['Bike_Model', 'Primary_Offence','Bike_Make', 'Location_Type', 'Report_DayOfYear','Occurrence_DayOfYear','Premises_Type' ,'Report_Hour','Bike_Speed', 'Cost_of_Bike','Status']]

# =============================================================================
# HANDLE MISSING VALUES -- Imputation
# =============================================================================
# replace Cost of Bike missing values with mean
data_group8['Cost_of_Bike'].fillna(data_group8['Cost_of_Bike'].mean(), inplace = True)

# replace Color of Bike missing values with not reported
# df['salary'] = df['salary'].fillna(df['salary'].mode()[0])
# data_group8['Bike_Colour'].fillna(data_group8['Bike_Colour'].mode()[0], inplace=True)

# replace Model of Bike missing values with not reported
# data_group8['Bike_Model'].fillna('NR', inplace = True)
data_group8['Bike_Model'].fillna(data_group8['Bike_Model'].mode()[0], inplace=True)


# =============================================================================
# CATEGORICAL DATA MANAGEMENT
# =============================================================================
# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels of feature column 

data_group8['Primary_Offence']= label_encoder.fit_transform(data_group8['Primary_Offence'])
data_group8['Location_Type']= label_encoder.fit_transform(data_group8['Location_Type']) 
data_group8['Premises_Type']= label_encoder.fit_transform(data_group8['Premises_Type']) 
data_group8['Bike_Make']= label_encoder.fit_transform(data_group8['Bike_Make']) 
data_group8['Bike_Model']= label_encoder.fit_transform(data_group8['Bike_Model']) 

# =============================================================================
# DATA NORMALIZATION
# =============================================================================
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
arr_bike_cost = scaler.fit_transform(data_group8[['Cost_of_Bike']]) # normalize cost of bike
df_norm = pd.DataFrame(arr_bike_cost, columns=['Bike_Cost']) # convert numpy array to dataframe

data_group8_normalized = df_norm.join(data_group8) # join normalized cost of bike with dummied
data_group8_normalized = data_group8_normalized.drop(['Cost_of_Bike'], axis=1)  # drop cost of bike original column and index column

# =============================================================================
# HANDLING IMBALANCED DATASET
# =============================================================================
count1 = (data_group8_normalized['Status'] == 1).sum() #check recovered count
count2 = (data_group8_normalized['Status'] == 0).sum() #check stolen count
print(count2)

# Up-sample minority class
from sklearn.utils import resample
# Separate majority and minority classes 
# minority is 1 --recovered Status
df_majority = data_group8_normalized[data_group8_normalized.Status==0]
df_minority = data_group8_normalized[data_group8_normalized.Status==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=29280,    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.Status.value_counts()

# Separate input features (X) and target variable (y)
y = df_upsampled.Status
X = df_upsampled.drop('Status', axis=1)

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================
from sklearn.model_selection import train_test_split
# split 30% for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('X_train: ' + str(len(X_train)))
print('X_test: ' + str(len(X_test)))
print('y_train: ' + str(len(y_train)))
print('y_test: ' + str(len(y_test)))
print(" ")

# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# # Train model
clf_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
clf_1.fit(X_train, y_train)
# Predict on training set
pred_y_1 = clf_1.predict(X_test)

# Is our model still predicting just one class?
print("Predictions: " + str(np.unique( pred_y_1 )))
# How's our accuracy?
print("LogisticRegression accuracy score: " + str (accuracy_score(y_test, pred_y_1) ))


# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(clf_1, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

# Report and confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, pred_y_1))
print(" ")
print("Classification Report: ")
print(classification_report(y_test, pred_y_1))


count2 = 0
for p in pred_y_1:
    if p == 1:
        count2 += 1

print('Total predictions: ' + str(pred_y_1.size))
print("Total bikes predicted 'recovered': " + str(count2))  

# =============================================================================
# DECISIONTREE CLASSIFIER
# =============================================================================
print(" ")
print(" ")
print(" ")
from sklearn.tree import DecisionTreeClassifier
clf_2 = DecisionTreeClassifier(max_depth=42, criterion = 'entropy', random_state=42)
clf_2.fit(X_train, y_train)
# Predict on training set
pred_y_2 = clf_2.predict(X_test)

# Is our model still predicting just one class?
print("Predictions: " + str(np.unique( pred_y_2 )))

# How's our accuracy?
print("DecisionTree accuracy score: " + str(accuracy_score(y_test, pred_y_2)))

# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(clf_2, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

# Report and confusion matrix
print("Confusion matrix:")
print(" ")
print("Classification Report: ")
print(confusion_matrix(y_test, pred_y_2))
print(classification_report(y_test, pred_y_2))

count = 0
for p in pred_y_2:
    if p == 1:
        count += 1

print('Total predictions: ' + str(pred_y_2.size))
print("Total bikes predicted 'recovered': " + str(count))    


# =============================================================================
# SERIALIZE MODEL
# =============================================================================
# Serialie
import joblib 
joblib.dump(clf_2, 'C:/Users/ivanz/Desktop/Data Warehouse/model_group8_2022.pkl')
print("Model dumped!")
#
model_columns = list(X.columns)
print(model_columns)
joblib.dump(model_columns, 'C:/Users/ivanz/Desktop/Data Warehouse/model_columns_group8_2022.pkl')
print("Models columns dumped!")
#



