from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('svm_model.pkl')

# Define a route for predictions
# @app.route('/predict', methods=['POST'])
@app.route('/')
def home():
    return "Welcome to the SVM Model API!,its me again, its getting tough,man"

@app.route('/about/<username>')
def about_page(username):
    return f'<h1>Hello {username}!</h1>'

@app.route('/predict', methods=['POST'])

def predict():
    # Get the JSON data from the request
    data = request.json
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
    app.run(debug=True)

