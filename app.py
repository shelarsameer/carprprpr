from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('carprpr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Home route (index)
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    inputs = [
        request.form['model'], 
        request.form['year'],
        request.form['transmission'],
        request.form['mileage'], 
        request.form['fuelType'],
        request.form['tax'], 
        request.form['mpg'],
        request.form['engineSize']
    ]

    # Convert the inputs to a numpy array
    features = np.array([inputs])

    # Apply any necessary preprocessing (if needed, e.g., scaling)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Predict the price using the loaded model
    predicted_price = model.predict(scaled_features)

    # Return the result
    return render_template('result.html', prediction=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
