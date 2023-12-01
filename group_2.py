import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

path = './data/'
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(path, filename)
dataFrame = pd.read_csv(fullpath)
print(dataFrame)

print(dataFrame.head())
print(dataFrame.columns.values)
print(dataFrame.shape)
print(dataFrame.describe())
print(dataFrame.dtypes)
print(dataFrame.head(5))

premises_counts = dataFrame.columns.values
print(premises_counts)

print("Data in PREMISES_TYPE column:")
print(dataFrame['PREMISES_TYPE'].head(10).tolist())


print("Counts of Different Values in PREMISES_TYPE:")
print(dataFrame['PREMISES_TYPE'].value_counts())

#fill the blank value in PRIMARY_OFFENCE to unknown
print(dataFrame['PRIMARY_OFFENCE'].value_counts())
dataFrame['PRIMARY_OFFENCE'] = dataFrame['PRIMARY_OFFENCE'].fillna('unknown')
print(dataFrame['PRIMARY_OFFENCE'].value_counts())

numeric_data = dataFrame.apply(pd.to_numeric, errors='coerce')
correlation_matrix = numeric_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plotting a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Matrix")
plt.show()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
dataFrame['LOCATION_TYPE'] = label_encoder.fit_transform(dataFrame['LOCATION_TYPE'])
dataFrame['STATUS'] = label_encoder.fit_transform(dataFrame['STATUS'])

# calculate the correlation between LOCATION_TYPE and STATUS
column1 = 'LOCATION_TYPE'
column2 = 'STATUS'
correlation_coefficient = dataFrame['STATUS'].corr(dataFrame['LOCATION_TYPE'])
# Create a scatter plot with a regression line
sns.regplot(x=column1, y=column2, data=dataFrame)
plt.title(f'Scatter Plot with Regression Line\nCorrelation: {correlation_coefficient:.2f}')
plt.show()

# Group by 'REPORT_YEAR', count the occurrences
yearly_theft_counts = dataFrame.groupby('REPORT_YEAR').size()

# Plot a bar chart for each year
plt.figure(figsize=(12, 6))
yearly_theft_counts.plot(kind='bar', color='blue')
plt.title('Total Bike Theft by Year')
plt.xlabel('Year')
plt.ylabel('Number of Thefts')
plt.show()

# Group by 'REPORT_MONTH', count the occurrences
monthly_theft_counts = dataFrame.groupby('REPORT_MONTH').size()
plt.figure(figsize=(12, 6))
monthly_theft_counts.plot(kind='bar', color='red')
plt.title('Bike Theft by Month')
plt.xlabel('Month')
plt.ylabel('Number of Thefts')
plt.show()

# Group by 'REPORT_HOUR', count the occurrences
hourly_theft_counts = dataFrame.groupby('REPORT_HOUR').size()
plt.figure(figsize=(12, 6))
hourly_theft_counts.plot(kind='bar', color='green')
plt.title('Bike Theft by Hour')
plt.xlabel('Hour')
plt.ylabel('Number of Thefts')
plt.show()