# -*- coding: utf-8 -*-
"""
Created on

@author: Group 2
"""

# Import necessary libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px

# Set path for the data
path = "D:/00 Centennial/01 Fall 2023/04 COMP309 - Data Warehouse & Predictive Anltcs/11 Group Project/COMP309GroupProject/"
dataPath = os.path.join(path, "data/")
filename = 'Bicycle_Thefts_Open_Data.csv'
data = os.path.join(dataPath, filename)

# Load data from a CSV file
data_group2 = pd.read_csv(data)

# Basic checks and information about the data
print(data_group2.head(5))  # First 5 rows
print(data_group2.shape)    # Rows and columns
print(data_group2.columns)  # Column names
print(data_group2.info())   # Data types and counts
print(data_group2.isnull().sum())  # Missing values

# Unique values in each column
for i in data_group2.columns:
    print(i, ' ', data_group2[i].nunique(dropna=True))

# Count of Stolen, Recovered, and Unknown Bicycles
print(data_group2['STATUS'].value_counts())

# Count types of location theft occurrences
print(data_group2['LOCATION_TYPE'].value_counts())

# Check/count neighborhood with highest #events
print(data_group2['NEIGHBOURHOOD_140'].value_counts())

# Count most stolen bike color
print(data_group2['BIKE_COLOUR'].value_counts())

# Count most stolen bike speed
print(data_group2['BIKE_SPEED'].value_counts())

# Count most stolen bike make
print(data_group2['BIKE_MAKE'].value_counts())

# Count most stolen bike model
print(data_group2['BIKE_MODEL'].value_counts())

################# STATISTICAL ASSESSMENTS ################
print(data_group2.describe(include='all'))

# Filter out only numeric columns for correlation
numeric_cols = data_group2.select_dtypes(include=[np.number])
corr = numeric_cols.corr()

# Generate heatmap for correlation
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_title('Bike Theft Correlation')
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, linewidth=0.5)
plt.show()

################# MISSING DATA EVALUATIONS ################
print(data_group2.isna().sum())

################# GRAPHS AND VISUALIZATIONS ################
pio.renderers.default = 'browser'

# Bar chart for Bicycle Status
df_status = data_group2['STATUS'].value_counts(normalize=False)
fig_status = px.bar(df_status, x=df_status.index, y=df_status)
fig_status.update_layout(
    xaxis_title='Status of Bicycle Theft',
    yaxis_title='Number of Events/Occurrences',
    title='Ratio of Bicycle Thefts Based on Status')
fig_status.show()

# Bar chart for Neighborhood with the Highest Event
df_neighbourhood = data_group2['NEIGHBOURHOOD_140'].value_counts(normalize=False).head(10)
fig_neighbourhood = px.bar(df_neighbourhood, x=df_neighbourhood.index, y=df_neighbourhood)
fig_neighbourhood.update_layout(
    xaxis_title='Neighbourhood',
    yaxis_title='Number of Theft Occurrences',
    title='Top 10 Neighbourhoods with Highest Number of Bicycle Thefts')
fig_neighbourhood.show()

# Bar chart for Location Type with the Highest Event
df_location_type = data_group2['LOCATION_TYPE'].value_counts().head(10)
fig_location_type = px.bar(df_location_type, x=df_location_type.index, y=df_location_type)
fig_location_type.update_layout(
    xaxis_title='Location Type',
    yaxis_title='Number of Theft Occurrences',
    title='Top 10 Location Types with Highest Number of Bicycle Thefts')
fig_location_type.show()

# Bar chart for Bicycle Colours with Highest Number of Theft
df_bike_colour = data_group2['BIKE_COLOUR'].value_counts().head(12)
fig_bike_colour = px.bar(df_bike_colour, x=df_bike_colour.index, y=df_bike_colour)
fig_bike_colour.update_layout(
    xaxis_title='Bicycle Colour',
    yaxis_title='Number of Thefts',
    title='Top 12 Bicycle Colours with Highest Number of Thefts')
fig_bike_colour.show()


# Filter data for non-null latitudes and longitudes
data_subset = data_group2[data_group2['BIKE_COST'].notnull() & data_group2['LAT_WGS84'].notnull() & data_group2['LONG_WGS84'].notnull()]

# Sample a subset of data to avoid overplotting
data_sample = data_subset.sample(n=100, random_state=1)

fig = px.scatter_mapbox(
    data_frame=data_sample,
    lat='LAT_WGS84',
    lon='LONG_WGS84',
    color='STATUS',
    size='BIKE_COST',
    zoom=10,
    mapbox_style="open-street-map"
)

fig.update_layout(title="Map of Bicycle Theft Status")
fig.show()

# # Interactive map of Bicycle Status with size based on Cost
# fig_map_status = px.scatter_mapbox(
#     data_frame=data_group2[data_group2['BIKE_COST'].notnull()],
#     lat='LAT_WGS84',
#     lon='LONG_WGS84',
#     color='STATUS',
#     size='BIKE_COST',
#     zoom=10)
# fig_map_status.update_layout(title="Map of Bicycle Theft Status", mapbox_style="carto-darkmatter")
# fig_map_status.show()

#
# import plotly.express as px
#
# # Example coordinates for a single point
# data = pd.DataFrame({
#     'LAT_WGS84': [43.637658],  # Replace with your latitude
#     'LONG_WGS84': [-79.443654],  # Replace with your longitude
#     'STATUS': ['STOLEN'],  # Replace with your status
#     'BIKE_COST': [1000]  # Replace with your bike cost
# })
#
# fig = px.scatter_mapbox(
#     data_frame=data,
#     lat='LAT_WGS84',
#     lon='LONG_WGS84',
#     color='STATUS',
#     size='BIKE_COST',
#     zoom=10,
#     mapbox_style="open-street-map"
# )
#
# fig.update_layout(title="Map of Bicycle Theft Status")
# fig.show()