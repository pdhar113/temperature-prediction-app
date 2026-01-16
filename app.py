from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("temperature_lr_model.pkl")
MODEL_CONFIDENCE = 0.70  # R² score from training

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    aqi = float(request.form["aqi"])
    humidity = float(request.form["humidity"])

    input_data = np.array([[aqi, humidity]])
    prediction = model.predict(input_data)[0]

    return render_template(
        "result.html",
        temperature=round(prediction, 2),
        confidence=round(MODEL_CONFIDENCE * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)

