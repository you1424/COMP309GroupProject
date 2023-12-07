# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 08:32:06 2022

@author: ivanz
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if clf_2:
        try:
            json_ = request.json
            print(json_)
            # query = pd.get_dummies(pd.DataFrame(json_))
            query = pd.DataFrame(json_)
            
            query['Primary_Offence']= label_encoder.fit_transform(query['Primary_Offence'])
            query['Location_Type']= label_encoder.fit_transform(query['Location_Type']) 
            query['Premises_Type']= label_encoder.fit_transform(query['Premises_Type']) 
            query['Bike_Make']= label_encoder.fit_transform(query['Bike_Make']) 
            query['Bike_Model']= label_encoder.fit_transform(query['Bike_Model']) 
            # query = query.reindex(columns=model_columns, fill_value=0)
            print(query)

            # from sklearn import preprocessing
            # scaler = preprocessing.StandardScaler()
            # # Fit your data on the scaler object
            # scaled_df = scaler.fit_transform(query)
            # # return to data frame
            # query = pd.DataFrame(scaled_df, columns=model_columns)
            # print(query)
            prediction = list(clf_2.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    clf_2 = joblib.load('C:/Users/ivanz/Desktop/Data Warehouse/model_group8_2022.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('C:/Users/ivanz/Desktop/Data Warehouse/model_columns_group8_2022.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
