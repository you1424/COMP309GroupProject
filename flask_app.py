import joblib
from flask import Flask, request, jsonify
import traceback
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)
label_encoder = preprocessing.LabelEncoder()

# Load the model outside the function
pkl_file = joblib.load('./pkl_files/model_group2_2023.pkl')  # Load pkl model
print('Model loaded')

'''
The root route
'''
@app.route('/')
def home():
    return 'This is the home page'

'''
Get and POST methods example
'''
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if pkl_file:
        try:
            # Get JSON data from the request
            json_data = request.json
            if not json_data:
                return jsonify({'error': 'No input data provided'})

            # Ensure that all required fields are present
            required_fields = ['STATUS', 'LOCATION_TYPE', 'NEIGHBOURHOOD_158', 'BIKE_COLOUR', 'BIKE_SPEED', 'BIKE_MAKE', 'BIKE_MODEL', 'BIKE_COST']
            for field in required_fields:
                if field not in json_data:
                    return jsonify({'error': f'Missing required field: {field}'})

            # Convert BIKE_SPEED to integer
            json_data['BIKE_SPEED'] = int(json_data['BIKE_SPEED'])

            # Create a DataFrame for prediction
            query = pd.DataFrame([json_data])

            # Make prediction
            prediction = list(pkl_file.predict(query))

            return jsonify({'prediction': str(prediction)})

        except Exception as e:
            return jsonify({'error': f'Something went wrong: {str(e)}'})
    else:
        return jsonify({'error': 'Model not available. Train the model first.'})

if __name__ == '__main__':
    app.run(debug=True)
