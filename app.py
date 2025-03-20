from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    chest_pain_type = int(request.form["chest_pain_type"])
    resting_bp = int(request.form["resting_bp"])
    cholesterol = int(request.form["cholesterol"])
    fasting_blood_sugar = int(request.form["fasting_blood_sugar"])
    resting_ecg = int(request.form["resting_ecg"])
    max_heart_rate = int(request.form["max_heart_rate"])
    exercise_angina = int(request.form["exercise_angina"])
    oldpeak = float(request.form["oldpeak"])
    ST_slope = int(request.form["ST_slope"])

    # Prepare input data for prediction
    input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, 
                            resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope]])

    # Make prediction
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
