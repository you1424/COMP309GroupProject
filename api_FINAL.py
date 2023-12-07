# -*- coding: utf-8 -*-
"""

"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
from pandas import json_normalize
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if clf_2:
        try:
            json_ = request.json
            print(json_)
            df = json_normalize(json_) 
            df = df.reindex(columns=model_columns, fill_value=0)
            
            
            label_encoder = LabelEncoder()
            df['Primary_Offence']= label_encoder.fit_transform(df['Primary_Offence'])
          
            df['Bike_Type'] = label_encoder.fit_transform(df['Bike_Type'])
            df['Bike_Make'] = label_encoder.fit_transform(df['Bike_Make'])

            print(df)
            prediction = list(clf_2.predict(df))
            print({'prediction': str(prediction)})
            count = 0
            count1 = 0
            for p in prediction:
                if p == 1:
                    count += 1
                if p ==0:
                    count1 += 1

            total = count + count1
            percent_recovered = count / total * 100
            print('Total number of bikes found: ' + str(count) + ' out of ' + str(total) + ' stolen')
            print('Rate of recovery: ' + str(round(percent_recovered, 2)) + '%')
            return jsonify({'prediction': str(prediction)})

            

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model firs')
        return ('No model here to us')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    clf_2 = joblib.load('pkl_files/model_group2_2023.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('pkl_files/model_group2_2023.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
