import joblib
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the model and LabelEncoder
model = joblib.load('./pkl_files/bike_color_prediction_model.pkl')
label_encoder = joblib.load('./pkl_files/bike_color_label_encoder.pkl')

@app.route('/')
def home():
    return 'This is the home page'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model:
        try:
            json_data = request.json
            if not json_data:
                return jsonify({'error': 'No input data provided'})

            required_fields = ['BIKE_COLOUR']
            for field in required_fields:
                if field not in json_data:
                    return jsonify({'error': f'Missing required field: {field}'})

            # Encode the bike color
            bike_color_encoded = label_encoder.transform([json_data['BIKE_COLOUR']])

            # Create a DataFrame for prediction
            query = pd.DataFrame({'BIKE_COLOUR': bike_color_encoded})

            # Make prediction
            prediction = model.predict(query)
            return jsonify({'prediction': str(prediction[0])})

        except Exception as e:
            return jsonify({'error': f'Something went wrong: {str(e)}'})
    else:
        return jsonify({'error': 'Model not available. Train the model first.'})

if __name__ == '__main__':
    app.run(debug=True)
