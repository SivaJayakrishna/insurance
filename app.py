import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])
    
    # Create input array for prediction
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return render_template('output.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)