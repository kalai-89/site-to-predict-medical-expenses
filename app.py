from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user inputs from the form
    age = float(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    hr_pulse = float(request.form['hr_pulse'])
    rr = float(request.form['rr'])
    gender_male = 1 if request.form['gender'].lower() == 'male' else 0
    marital_status_unmarried = 1 if request.form['marital_status'].lower() == 'unmarried' else 0
    key_complaints_code_cad = 1 if request.form['key_complaints_code'].lower() == 'cad' else 0
    key_complaints_code_os_asd = 1 if request.form['key_complaints_code'].lower() == 'os-asd' else 0
    key_complaints_code_rhd = 1 if request.form['key_complaints_code'].lower() == 'rhd' else 0
    key_complaints_code_other = 1 if request.form['key_complaints_code'].lower() == 'other' else 0

    # Prepare input features as a numpy array
    features = np.array([[age, weight, height, hr_pulse, rr, gender_male, marital_status_unmarried,
                          key_complaints_code_cad, key_complaints_code_os_asd, key_complaints_code_rhd,
                          key_complaints_code_other, 0]])  # Add a placeholder for the missing feature

    # Make prediction
    predicted_expense = model.predict(features)

    return render_template('result.html', predicted_expense=predicted_expense[0])

if __name__ == '__main__':
    app.run(debug=True)
