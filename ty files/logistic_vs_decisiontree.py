# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:13:31 2023

@author: tyrel
"""

##### Load Data #####
print("LOADING DATA...")
import pandas as pd
import os
#path = "C:/A_COMP309/data/Datasets for Predictive Modelling/Datasets for Predictive
path = "F:/_FALL 2023/COMP309/Bicycle Theft Project/"
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(path,filename)
data_group2 = pd.read_csv(fullpath)
#print(data_group2.columns)

# Remove UNKNOWN's from Status column
data_group2 = data_group2.loc[data_group2["STATUS"] != 'UNKNOWN']
# Reference Status column to 0 or 1
data_group2['STATUS'].replace(['STOLEN', 'RECOVERED'],[0, 1], inplace=True)

# columns chosen from the toronto data set:
data_group2 = data_group2[['BIKE_MODEL', 'PRIMARY_OFFENCE','BIKE_MAKE', 'LOCATION_TYPE', 'REPORT_DOY',
                           'OCC_DOY','PREMISES_TYPE' ,'REPORT_HOUR','BIKE_SPEED', 'BIKE_COST','STATUS']]

##### Imputation (process blank and NaN values) #####
print("FIXING BLANK VALUES...")

# replace BIKE_MODEL blank values with MODE <--- most occuring bike model
data_group2['BIKE_MODEL'].fillna(data_group2['BIKE_MODEL'].mode()[0], inplace=True)

# replace BIKE_COST blank values with mean
data_group2['BIKE_COST'].fillna(data_group2['BIKE_COST'].mean(), inplace = True)

# Handle other columns with missing values using an imputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', or 'most_frequent'

# Impute missing values in numerical columns
numerical_columns = ['REPORT_DOY', 'OCC_DOY', 'REPORT_HOUR', 'BIKE_SPEED']
data_group2[numerical_columns] = imputer.fit_transform(data_group2[numerical_columns])

#### Categorical Data Management ####
from sklearn import preprocessing # Import label encoder 
label_encoder = preprocessing.LabelEncoder()

# Encode labels of feature column 
data_group2['PRIMARY_OFFENCE']= label_encoder.fit_transform(data_group2['PRIMARY_OFFENCE'])
data_group2['LOCATION_TYPE']= label_encoder.fit_transform(data_group2['LOCATION_TYPE']) 
data_group2['PREMISES_TYPE']= label_encoder.fit_transform(data_group2['PREMISES_TYPE']) 
data_group2['BIKE_MAKE']= label_encoder.fit_transform(data_group2['BIKE_MAKE']) 
data_group2['BIKE_MODEL']= label_encoder.fit_transform(data_group2['BIKE_MODEL']) 

#### Data Normalization ####
print("NORMALIZING DATA...")
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
arr_bike_cost = scaler.fit_transform(data_group2[['BIKE_COST']]) # normalize cost of bike
df_norm = pd.DataFrame(arr_bike_cost, columns=['Bike_Cost']) # convert numpy array to dataframe

data_group2_normalized = df_norm.join(data_group2) # join normalized cost of bike with dummied
data_group2_normalized = data_group2_normalized.drop(['BIKE_COST'], axis=1)  # drop cost of bike original column and index column

#### Handling Imbalanced Dataset ####
print("HANDLING IMBALANCED DATASET...")
count1 = (data_group2_normalized['STATUS'] == 1).sum() #check recovered count
count2 = (data_group2_normalized['STATUS'] == 0).sum() #check stolen count
print("   normalized RECOVERED =", count1)
print("   normalized STOLEN = ", count2)

# Upsample minority class
from sklearn.utils import resample
# Separate majority and minority classes (minority is 1 --> recovered Status)
df_majority = data_group2_normalized[data_group2_normalized.STATUS==0]
df_minority = data_group2_normalized[data_group2_normalized.STATUS==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True,     # sample with replacement
                                 n_samples=29280,    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.STATUS.value_counts()

# Separate input features (X) and target variable (y)
y = df_upsampled.STATUS
X = df_upsampled.drop('STATUS', axis=1)

#### Splitting Data ####
print("SPLITTING DATA (20/80 Split)...")
from sklearn.model_selection import train_test_split
# split 30% for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('   X_train: ' + str(len(X_train)))
print('   X_test: ' + str(len(X_test)))
print('   y_train: ' + str(len(y_train)))
print('   y_test: ' + str(len(y_test)))

#### Logistic Regression ####
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Separate input features (X) and target variable (y) for the logistic regression model
y = data_group2['STATUS']
X = data_group2.drop('STATUS', axis=1)

# Split the data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in the training and test data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train model
print("LOGISTIC REGRESSION START...")
clf_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
clf_1.fit(X_train_imputed, y_train)

# Predict on the test set
pred_y_1 = clf_1.predict(X_test_imputed)

# Is our model still predicting just one class?
print("   Predictions: " + str(np.unique( pred_y_1 )))
# How's our accuracy?
print("   Logistic Regression accuracy: " + str (accuracy_score(y_test, pred_y_1) ))

print("10 FOLD CROSS VALIDATION START...")
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(clf_1, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('   10 fold run score: ',score)

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

print('   Total predictions: ' + str(pred_y_1.size))
print("   Total bikes predicted 'recovered': " + str(count2))


#### Decision Tree ####
print("DECISION TREE START...")
from sklearn.tree import DecisionTreeClassifier

clf_2 = DecisionTreeClassifier(max_depth=42, criterion = 'entropy', random_state=42)
#clf_2.fit(X_train, y_train)
clf_2.fit(X_train_imputed, y_train)


# Predict on the test set
#pred_y_2 = clf_2.predict(X_test)
pred_y_2 = clf_2.predict(X_test_imputed)

# Is our model still predicting just one class?
print("   Predictions: " + str(np.unique( pred_y_2 )))

# How's our accuracy?
print("   DecisionTree accuracy: " + str(accuracy_score(y_test, pred_y_2)))

# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(clf_2, X_train, y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('   10 fold run score: ',score)

# Report and confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, pred_y_2))
print(" ")
print("Classification Report: ")
print(classification_report(y_test, pred_y_2))

count = 0
for p in pred_y_2:
    if p == 1:
        count += 1

print('   Total predictions: ' + str(pred_y_2.size))
print("   Total bikes predicted 'recovered': " + str(count))    

#### Dumping pkl Files ####
print("CREATING PKL FILES START...")
import joblib 

joblib.dump(clf_1, 'F:/_FALL 2023/COMP309/Bicycle Theft Project/pkl_files/grp2_logisticregressionmodel.pkl')
print("   Logistic Regression Model dumped!")


joblib.dump(clf_2, 'F:/_FALL 2023/COMP309/Bicycle Theft Project/pkl_files/grp2_decisiontreemodel.pkl')
print("   Decision Tree Model dumped!")

model_columns = list(X.columns)
#print(model_columns)
joblib.dump(model_columns, 'F:/_FALL 2023/COMP309/Bicycle Theft Project/pkl_files/grp2_model_columns.pkl')
print("   Model columns dumped!")












