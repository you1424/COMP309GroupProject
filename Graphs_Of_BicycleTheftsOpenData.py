import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path =  "C:/Sorce Code/Centennial/s5/309/COMP309GroupProject/Bicycle_Thefts_Open_Data.csv"  # 実際のファイルパスに置き換える

data = pd.read_csv(file_path)  

#1 Bicycle Recovery Probability by Status
status_counts = data['STATUS'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
plt.title('The Current Status of Theft Incidents')
plt.show()

#2 Distribution by Major Crime Types (Pie Chart)
primary_offence_counts = data['PRIMARY_OFFENCE'].value_counts().nlargest(10)
plt.figure(figsize=(10, 10))
plt.pie(primary_offence_counts, labels=primary_offence_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution by Major Crime Types')
plt.show()


#3 Comparison of Crime Occurrence by Region (Bar Chart)
division_counts = data['DIVISION'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=division_counts.index, y=division_counts.values, palette='viridis')
plt.xlabel('Police Division')
plt.ylabel('Number of Incidents')
plt.title('Comparison of Crime Occurrence by Region')
plt.xticks(rotation=45)
plt.show()

#4 Comparison of Crime Occurrence by Region (Bar Chart)
division_counts = data['DIVISION'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=division_counts.index, y=division_counts.values, palette='viridis')
plt.xlabel('Police Division')
plt.ylabel('Number of Incidents')
plt.title('Comparison of Crime Occurrence by Region')
plt.xticks(rotation=45)
plt.show()



#5 Motorcycle occurrence map by region-------------------------------------------------------------
#------------------------------------------------------------------------------------------------
import folium
from folium import plugins
import seaborn as sns

file_path = "C:/Sorce Code/Centennial/s5/309/COMP309GroupProject/Bicycle_Thefts_Open_Data.csv"
data = pd.read_csv(file_path)

# Comparison of Crime Occurrence by Neighbourhood (Bar Chart)
neighbourhood_counts = data['NEIGHBOURHOOD_158'].value_counts()

# Folium initialize map
m = folium.Map(location=[43.70, -79.42], zoom_start=11)

# generate barchart
bar_chart = folium.plugins.MarkerCluster().add_to(m)
for neighbourhood, count in zip(neighbourhood_counts.index, neighbourhood_counts.values):
    # 地域ごとの中央座標を使用
    location = [data.loc[data['NEIGHBOURHOOD_158'] == neighbourhood, 'LAT_WGS84'].mean(),
                data.loc[data['NEIGHBOURHOOD_158'] == neighbourhood, 'LONG_WGS84'].mean()]
    
    folium.Marker(
        location=location,
        icon=None,
        popup=f'<strong>{neighbourhood}</strong><br>{count} Incidents',
        tooltip=f'{neighbourhood}: {count}',
    ).add_to(bar_chart)

# show map
m.save('crime_occurrence_by_neighbourhood_map.html')
#----------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------



#6 Monthly Crime Incidents Trend (Line Chart)
monthly_trend = data.groupby('OCC_MONTH')['OBJECTID'].count()
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_trend.index, y=monthly_trend.values, marker='o', color='orange')
plt.xlabel('Month of Occurrence')
plt.ylabel('Number of Incidents')
plt.title('Monthly Crime Incidents Trend')
plt.show()


#7 Distribution of Reported Incidents by Day of the Week (Bar Chart)
report_dow_counts = data['REPORT_DOW'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=report_dow_counts.index, y=report_dow_counts.values, palette='Set2')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Incidents')
plt.title('Distribution of Reported Incidents by Day of the Week')
plt.show()

#8 Distribution of Crime Incidents by Hour of the Day (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(data['OCC_HOUR'], bins=24, color='skyblue', edgecolor='black')
plt.xlabel('Hour of Occurrence')
plt.ylabel('Number of Incidents')
plt.title('Distribution of Crime Incidents by Hour of the Day')
plt.show()

#9 Comparison of Crime Incidents by Region (Bar Chart)
plt.figure(figsize=(12, 6))
sns.barplot(x='DIVISION', y='OBJECTID', data=data, palette='coolwarm')
plt.xlabel('Police Division')
plt.ylabel('Number of Incidents')
plt.title('Comparison of Crime Incidents by Region')
plt.xticks(rotation=45)
plt.show()

#10 Comparison of Crime Occurrence by Bicycle Status (Bar Chart)
status_counts = data['STATUS'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=status_counts.index, y=status_counts.values, palette='muted')
plt.xlabel('Bicycle Status')
plt.ylabel('Number of Incidents')
plt.title('Comparison of Crime Occurrence by Bicycle Status')
plt.show()

#11 Monthly Trend of Major Crime Types (Multiple Line Chart)
monthly_trend_by_offence = data.groupby(['OCC_MONTH', 'PRIMARY_OFFENCE'])['OBJECTID'].count().unstack()
plt.figure(figsize=(12, 8))
monthly_trend_by_offence.plot(marker='o', colormap='tab10')
plt.xlabel('Month of Occurrence')
plt.ylabel('Number of Incidents')
plt.title('Monthly Trend of Major Crime Types')
plt.legend(title='Primary Offence Type', bbox_to_anchor=(1, 1))
plt.show()

#12 Distribution of Crime Locations by Major Offence Type (Stacked Bar Chart)
location_type_counts = data.groupby(['LOCATION_TYPE', 'PRIMARY_OFFENCE'])['OBJECTID'].count().unstack()
plt.figure(figsize=(12, 8))
location_type_counts.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Location Type')
plt.ylabel('Number of Incidents')
plt.title('Distribution of Crime Locations by Major Offence Type')
plt.legend(title='Primary Offence Type', bbox_to_anchor=(1, 1))
plt.show()

#13 Heatmap of Reported Crimes by Day of the Week and Hour (Heatmap)
heatmap_data = data.pivot_table(index='REPORT_DOW', columns='OCC_HOUR', values='OBJECTID', aggfunc='count')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='g')
plt.xlabel('Hour of Occurrence')
plt.ylabel('Day of the Week')
plt.title('Heatmap of Reported Crimes by Day of the Week and Hour')
plt.show()

#14 Regional Distribution of Major Crime Categories (Grouped Bar Chart)
regional_distribution = data.groupby(['DIVISION', 'PRIMARY_OFFENCE'])['OBJECTID'].count().unstack()
plt.figure(figsize=(14, 8))
regional_distribution.plot(kind='bar', colormap='Paired')
plt.xlabel('Police Division')
plt.ylabel('Number of Incidents')
plt.title('Regional Distribution of Major Crime Categories')

#15 Number of incidents by LOCATION_TYPE
location_type_counts = data['LOCATION_TYPE'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=location_type_counts.index, y=location_type_counts.values, palette='viridis')
plt.xlabel('Location Type')
plt.ylabel('Number of Incidents')
plt.title('Incident Occurrence by Location Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#16 Number of incidents by PREMISES_TYPE
premises_type_counts = data['PREMISES_TYPE'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=premises_type_counts.index, y=premises_type_counts.values, palette='viridis')
plt.xlabel('Premises Type')
plt.ylabel('Number of Incidents')
plt.title('Incident Occurrence by Premises Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#17 Heat map of number of bicycle theft incidents by time of day and location
heatmap_data = data.groupby(['OCC_HOUR', 'LOCATION_TYPE'])['OBJECTID'].count().unstack()
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='g')
plt.xlabel('Location Type')
plt.ylabel('Hour of Occurrence')
plt.title('Heatmap of Incident Occurrence by Hour and Location Type')
plt.show()

#18 Incident Occurrence by Bike Model (Top 10)
bike_model_counts = data['BIKE_MODEL'].value_counts().nlargest(10) 
plt.figure(figsize=(12, 8))
sns.barplot(x=bike_model_counts.index, y=bike_model_counts.values, palette='viridis')
plt.xlabel('Bike Model')
plt.ylabel('Number of Incidents')
plt.title('Incident Occurrence by Bike Model (Top 10)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



