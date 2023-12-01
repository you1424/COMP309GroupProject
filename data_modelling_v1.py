import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


# Set file path
path = "D:/00 Centennial/01 Fall 2023/04 COMP309 - Data Warehouse & Predictive Anltcs/11 Group Project/"
dataPath = os.path.join(path, "data/")
pklPath = os.path.join(path, "pkl_files/")

filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(dataPath, filename)

# Load data
pdata = pd.read_csv(fullpath)
data_group2 = pd.DataFrame(pdata)

# Data cleaning and preprocessing
data_group2.drop(columns=['OBJECTID', 'EVENT_UNIQUE_ID', 'OCC_DATE', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'OCC_DAY',
                          'OCC_DOY', 'OCC_HOUR', 'REPORT_DATE', 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DOW',
                          'REPORT_DAY', 'REPORT_DOY', 'REPORT_HOUR', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140',
                          'NEIGHBOURHOOD_140', 'LONG_WGS84', 'LAT_WGS84'], inplace=True, errors='ignore')

# Handle non-numeric values in 'DIVISION' column
data_group2['DIVISION'] = data_group2['DIVISION'].str.replace('D', '')
data_group2['DIVISION'].replace('NSA', np.nan, inplace=True)  # Replace 'NSA' with NaN
data_group2.dropna(subset=['DIVISION'], inplace=True)  # Remove rows with NaN in 'DIVISION'
data_group2['DIVISION'] = data_group2['DIVISION'].astype(int)

# Convert Status to binary
data_group2['STATUS'].replace(['STOLEN', 'RECOVERED', 'UNKNOWN'], [0, 1, np.nan], inplace=True)
data_group2.dropna(subset=['STATUS'], inplace=True)  # Remove rows where STATUS is NaN

# Normalize 'BIKE_COST'
scaler = MinMaxScaler()
data_group2['BIKE_COST'] = scaler.fit_transform(data_group2[['BIKE_COST']]).round(5)

# Handle missing values for other features
for col in data_group2.columns:
    if data_group2[col].dtype == 'object':
        data_group2[col].fillna(data_group2[col].mode()[0], inplace=True)
    else:
        data_group2[col].fillna(data_group2[col].mean(), inplace=True)

# Select features
y = data_group2['STATUS']
X = data_group2.drop(['STATUS'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Preprocessing pipeline
categorical_cols = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model pipeline
model = IMBPipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE()),
        ('classifier', DecisionTreeClassifier())
    ])

model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy Score:', round(accuracy * 100, 2), '%')

# Save model and columns
model_filename = 'model_group2_2023.pkl'
model_fullpath = os.path.join(pklPath, model_filename)
joblib.dump(model, model_fullpath)
print(f"Model saved to {model_fullpath}!")

# Data Visualization
# Get feature names after OneHotEncoding
onehot_columns = list(
    model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols))
numeric_features = list(X.select_dtypes(include=['int64', 'float64']).columns)
feature_names = numeric_features + onehot_columns

# Plotting feature importance
feature_importances = model.named_steps['classifier'].feature_importances_
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.show()
