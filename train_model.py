import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("data/aqi_temp_humidity_normal_highcorr.csv")

# Features and target
X = df[["AQI", "Humidity"]]
y = df["Temperature"]

# Split dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create dataframes with features and target for saving
train_df = X_train.copy()
train_df["Temperature"] = y_train
test_df = X_test.copy()
test_df["Temperature"] = y_test

# Save split data to CSV files
train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)
print("Split data saved: train_data.csv (70%) and test_data.csv (30%)")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully")
print("R² Score (Confidence Level):", r2)

# Save model
joblib.dump(model, "temperature_lr_model.pkl")
print("Model saved as temperature_lr_model.pkl")
