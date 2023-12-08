import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# =============================================================================
# Load Data
# =============================================================================
path = "D:/00 Centennial/01 Fall 2023/04 COMP309 - Data Warehouse & Predictive Anltcs/11 Group Project/COMP309GroupProject/"
dataPath = os.path.join(path, "data/")
pklPath = os.path.join(path, "pkl_files/")
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(dataPath, filename)
data_group2 = pd.read_csv(fullpath)

# =============================================================================
# Data Cleaning and Preprocessing
# =============================================================================
# Keep only the bike color and status columns
data_group2 = data_group2[['BIKE_COLOUR', 'STATUS']]

# Encode the 'BIKE_COLOUR' column if it's categorical
label_encoder = LabelEncoder()
if data_group2['BIKE_COLOUR'].dtype == 'object':
    data_group2['BIKE_COLOUR'] = label_encoder.fit_transform(data_group2['BIKE_COLOUR'])

# =============================================================================
# Train-Test Split
# =============================================================================
y = data_group2['STATUS']  # Target variable
X = data_group2[['BIKE_COLOUR']]  # Feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# =============================================================================
# Model Training and Evaluation
# =============================================================================
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy Score:', round(accuracy * 100, 2), '%')

# =============================================================================
# Serialize Model and LabelEncoder
# =============================================================================
model_filename = 'bike_color_prediction_model.pkl'
model_fullpath = os.path.join(pklPath, model_filename)
joblib.dump(model, model_fullpath)
print(f"Model saved to {model_fullpath}!")

label_encoder_filename = 'bike_color_label_encoder.pkl'
label_encoder_fullpath = os.path.join(pklPath, label_encoder_filename)
joblib.dump(label_encoder, label_encoder_fullpath)
print(f"Label Encoder saved to {label_encoder_fullpath}!")
