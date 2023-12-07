# -*- coding: utf-8 -*-
"""

"""

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

X = data_group2.drop(['Status'],axis=1)
y = data_group2['Status'].values

# =============================================================================
# HEATMAP
# =============================================================================
#get correlations of each features in dataset
corrmat = data_group2.corr()
top_corr_features = corrmat.index
plt.title("Bike Theft Heatmap")
plt.figure(figsize=(50,50))

#plot heat map
g=sns.heatmap(data_group2[top_corr_features].corr(),annot=True,cmap="RdYlGn")