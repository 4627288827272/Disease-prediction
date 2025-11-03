from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("disease_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    data = [float(x) for x in request.form.values()]
    input_data = np.array(data).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    result = "Positive (Disease Detected)" if prediction == 1 else "Negative (No Disease)"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
