from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# Route to display the HTML form
@app.route('/')
def index():
    return render_template('form.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve data from form
    name = request.form.get('name')
    age = request.form.get('age')
    email = request.form.get('email')
    
    # Process or display the data as needed
    return f"Hello, {name}! You are {age} years old, and your email is {email}."

@app.route('/')
def home():
    return "Welcome to the SVM Model API!,its me again, its getting tough,man"

@app.route('/about/<username>')
def about_page(username):
    return f'<h1>Hello {username}!</h1>,<h3> How is the going, I LOVE YOU</h3?'

 
model = joblib.load("svm_model.pkl")
@app.route('/predict', methods=['POST'])

def predict():
    # Get the JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    features = np.array(data['selected_features_list']).reshape(1, -1)
    
    # Predict the class and probabilities
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    # Return the prediction and probabilities as a JSON response
    return jsonify({
        'prediction': int(prediction[0]),
        'probabilities': probabilities.tolist()
    })


    

if __name__ == '__main__':
    app.run( debug=True)

