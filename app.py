import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data = np.array(data).reshape(1, -1)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("diabetes_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    output = "Diabetic" if prediction == 1 else "Not Diabetic"

    return render_template('index.html', prediction_text=f'The person is {output}.')

if __name__ == "__main__":
    app.run(debug=True)
