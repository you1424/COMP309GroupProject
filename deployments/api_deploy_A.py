# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:51:43 2022

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
    if model:
        try:
            json_ = request.json
            print(json_)
            # query = pd.get_dummies(pd.DataFrame(json_))
            query = pd.DataFrame(json_)
            
            query['Division'] = query['Division'].str.replace('D','')
            query['Division'] = query['Division'].astype('int')
            
            query['Primary_Offence']= label_encoder.fit_transform(query['Primary_Offence'])
            query['Bike_Colour'] = label_encoder.fit_transform(query['Bike_Colour'])
            # query = query.reindex(columns=model_columns, fill_value=0)
            print(query)

            prediction = list(model.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

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

    model = joblib.load('C:/Users/ivanz/Desktop/Data Warehouse/model_group8_2022_ver2.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('C:/Users/ivanz/Desktop/Data Warehouse/model_columns_group8_2022_ver2.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
