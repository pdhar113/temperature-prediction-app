import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("data/aqi_temp_humidity_normal_highcorr.csv")

X = df[["AQI", "Humidity"]]
y = df["Temperature"]

# Load trained model
model = joblib.load("temperature_lr_model.pkl")

# Predict
predictions = model.predict(X)

# Error calculation
mse = mean_squared_error(y, predictions)

print("Testing completed")
print("Mean Squared Error:", mse)

# Sample prediction
sample = [[120, 60]]
predicted_temp = model.predict(sample)
print("Sample Prediction (AQI=120, Humidity=60):", predicted_temp[0])
