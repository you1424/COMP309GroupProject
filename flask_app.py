from flask import Flask, request, jsonify
from pandas import json_normalize

app = Flask(__name__)

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
    if request.method == 'GET':
        json_message = request.json
        print(json_message)
        return 'This is the GET method for the /predict route'
    elif request.method == 'POST':
        json_message = request.json
        print(json_message)
        return 'This is the POST method for the /predict route'

if __name__ == '__main__':
    app.run(debug=True)
