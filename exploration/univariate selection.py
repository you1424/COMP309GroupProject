# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt 

# Set path for the data
path = "D:/00 Centennial/01 Fall 2023/04 COMP309 - Data Warehouse & Predictive Anltcs/11 Group Project/COMP309GroupProject/"
dataPath = os.path.join(path, "data/")
filename = 'Bicycle_Thefts_Open_Data.csv'
data = os.path.join(dataPath, filename)

# Load data from a CSV file
data_group2 = pd.read_csv(data)

############# HANDLE MISSING VALUES -- Imputation #############
# replace Cost of Bike missing values with mean
data_group2['Cost_of_Bike'].fillna(data_group2['Cost_of_Bike'].mean(), inplace = True)

# replace Color of Bike missing values with not reported
# df['salary'] = df['salary'].fillna(df['salary'].mode()[0])
data_group2['Bike_Colour'].fillna(data_group2['Bike_Colour'].mode()[0], inplace=True)

#------- fill the null values in the 'Color' column with 'Not Reported'
data_group2['Bike_Colour'].fillna('Not reported', inplace = True)

# Import label encoder 
from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data_group2['event_unique_id']= label_encoder.fit_transform(data_group2['event_unique_id']) 
data_group2['Occurrence_Month']= label_encoder.fit_transform(data_group2['Occurrence_Month']) 
data_group2['Primary_Offence']= label_encoder.fit_transform(data_group2['Primary_Offence'])
data_group2['Occurrence_DayOfWeek']= label_encoder.fit_transform(data_group2['Occurrence_DayOfWeek']) 
data_group2['Report_Month']= label_encoder.fit_transform(data_group2['Report_Month']) 
data_group2['Report_DayOfWeek']= label_encoder.fit_transform(data_group2['Report_DayOfWeek']) 
data_group2['Division']= label_encoder.fit_transform(data_group2['Division']) 
data_group2['City']= label_encoder.fit_transform(data_group2['City']) 
data_group2['Hood_ID']= label_encoder.fit_transform(data_group2['Hood_ID']) 
data_group2['NeighbourhoodName']= label_encoder.fit_transform(data_group2['NeighbourhoodName']) 
data_group2['Location_Type']= label_encoder.fit_transform(data_group2['Location_Type']) 
data_group2['Premises_Type']= label_encoder.fit_transform(data_group2['Premises_Type']) 
data_group2['Bike_Make']= label_encoder.fit_transform(data_group2['Bike_Make']) 
data_group2['Bike_Model']= label_encoder.fit_transform(data_group2['Bike_Model']) 
data_group2['Bike_Type']= label_encoder.fit_transform(data_group2['Bike_Type']) 
data_group2['Bike_Colour']= label_encoder.fit_transform(data_group2['Bike_Colour']) 
data_group2['Status']= label_encoder.fit_transform(data_group2['Status']) 


X = data_group2.drop(['Status','Occurrence_Date','Report_Date','X','Longitude'],axis=1) # drop dates and negative value --solve error
y = data_group2['Status'].values


# =============================================================================
# UNIVARIATE SELECTION
# =============================================================================
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(15,'Score'))  #print 10 best features