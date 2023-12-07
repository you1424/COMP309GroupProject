#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# =============================================================================
# LOAD DATA
# =============================================================================
import os
path = "D:/00 Centennial/01 Fall 2023/04 COMP309 - Data Warehouse & Predictive Anltcs/11 Group Project/"
dataPath = os.path.join(path, "data/")
pklPath = os.path.join(path, "pkl_files/")
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(path, filename)
data_group8 = pd.read_csv(fullpath)

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# Check number or rows and columns
print(data_group8.shape)

#-----------remove entries where the theft was outside of Toronto
data_group8 = data_group8[data_group8['City'] == 'Toronto']


#--------Remove columns that we will not be using for our data modeling--------------
data_group8 = data_group8[['Occurrence_Date', 'Primary_Offence', 'Report_Date',
                           'Division', 'Bike_Colour', 'Cost_of_Bike', 'Status' ]]


#remove 'D' from Division column so that it is numeric only and convert it to an int type object
data_group8['Division'] = data_group8['Division'].str.replace('D','')
data_group8['Division'] = data_group8['Division'].astype('int')


#----------Transform all date columns in the Dataframe to datetime dtype objects
data_group8['Occurrence_Date'] = data_group8['Occurrence_Date'].str[:10]
data_group8['Occurrence_Date'] = data_group8['Occurrence_Date'].apply(pd.to_datetime)

data_group8['Report_Date'] = data_group8['Report_Date'].str[:10]
data_group8['Report_Date'] = data_group8['Report_Date'].apply(pd.to_datetime)

#------------subtract occurance date from report date to find out how many days the bike wen unreported

data_group8['DaysBeforeReporting'] = (data_group8['Report_Date'] - data_group8['Occurrence_Date']).dt.days

# =============================================================================
# HANDLE MISSING VALUES -- Imputation
# =============================================================================
#------ remove entires from the dataframes where the bike's status is unknown
data_group8 = data_group8[data_group8['Status'] != 'UNKNOWN']

# Reference Status column to 0 or 1
data_group8['Status'].replace(['STOLEN', 'RECOVERED'],[0, 1], inplace=True)

#------- fill the null values in the 'Cost-of_Bike' columns with the mean
data_group8['Cost_of_Bike'].fillna(data_group8['Cost_of_Bike'].mean(), inplace = True)

# =============================================================================
# DATA NORMALIZATION
# =============================================================================
#------- normalize the values in the 'Cost-of-Bike' column
scaler = MinMaxScaler()
arr_bike_cost = scaler.fit_transform(data_group8[['Cost_of_Bike']])
data_group8['Cost_of_Bike'] = arr_bike_cost.round(5)

#------- fill the null values in the 'Color' column with 'Not Reported'
data_group8['Bike_Colour'].fillna('Not reported', inplace = True)

# =============================================================================
# CATEGORICAL DATA MANAGEMENT
# =============================================================================
from sklearn.preprocessing import LabelEncoder

#create instance of label encoder
label_encoder = LabelEncoder()
#perform label encoding on remaining categorical columns
data_group8['Primary_Offence']= label_encoder.fit_transform(data_group8['Primary_Offence'])
data_group8['Bike_Colour'] = label_encoder.fit_transform(data_group8['Bike_Colour'])


#drop Occurrence_Data and Report_Date as the significant data was extrated into the DaysBefore Reporting column
cleaned_df = data_group8.drop(['Occurrence_Date', 'Report_Date'], axis=1)

#get correlations of each features in dataset
corrmat = cleaned_df.corr()
corrmat.style.background_gradient(cmap='coolwarm')
top_corr_features = corrmat.index
fig, ax = plt.subplots(figsize=(20, 10))
plt.title("Bike Theft Heatmap")
plt.figure(figsize=(50,50))

#plot heat map
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(corrmat, dtype=bool))
g = sns.heatmap(corrmat, annot=True,  mask = mask, cmap=cmap, linewidth=0.5)
#g=sns.heatmap(cleaned_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# separating features from target
y = cleaned_df['Status'].values # target
X = cleaned_df.drop(['Status'],axis=1) # features

# =============================================================================
# HANDLING IMBALANCED DATASET
# =============================================================================
# Up-sample minority class
from sklearn.utils import resample
# Separate majority and minority classes
# minority is 1 --recovered Status
df_majority = cleaned_df[cleaned_df.Status==0]
df_minority = cleaned_df[cleaned_df.Status==1]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =============================================================================
# DECISIONTREE CLASSIFIER
# =============================================================================
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)


print(cleaned_df.head(10))
#print(data_group8.max())
print(cleaned_df.dtypes)
##print(data_group8.info())
#print(len(train))
#print(len(test))

print('Accuracy Score:' + str((score * 100).round(2)) + '%')

count = 0
for p in predictions:
    if p == 1:
        count += 1

print('Total predictions: ' + str(predictions.size))
print("Total bikes predicted 'recovered': " + str(count))
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# =============================================================================
# SERIALIZE MODEL
# =============================================================================

# Save model and columns
model_filename = 'model2080_group2_2023.pkl'
model_fullpath = os.path.join(pklPath, model_filename)
joblib.dump(model, model_fullpath)
print(f"Model saved to {model_fullpath}!")

#
model_columns_filename = 'model2080_group2_2023.pkl'
model_columns_fullpath = os.path.join(pklPath, model_columns_filename)
model_columns = list(X.columns)
print(model_columns)
joblib.dump(model, model_columns_fullpath)
print(f"Model Columns saved to {model_fullpath}!")
#



