# ML Lab: Temperature Prediction

A machine learning application that predicts temperature based on Air Quality Index (AQI) and Humidity levels using Linear Regression.

## Project Overview

This project demonstrates a complete ML pipeline from data preprocessing, model training, and evaluation to a Flask-based web application for making predictions.

**Model Performance:**
- R² Score: 0.70 (70% confidence level)
- Algorithm: Linear Regression
- Features: AQI, Humidity
- Target: Temperature

## Project Structure

```
ml_lab/
├── app.py                              # Flask web application
├── train_model.py                      # Model training script
├── test_model.py                       # Model testing/evaluation
├── temperature_lr_model.pkl            # Trained model (generated)
├── data/
│   ├── aqi_temp_humidity_normal_highcorr.csv   # Original dataset
│   ├── train_data.csv                  # 70% training data (generated)
│   └── test_data.csv                   # 30% testing data (generated)
├── templates/
│   ├── index.html                      # Input form page
│   └── result.html                     # Prediction results page
└── README.md                           # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone/Navigate to the project directory:**
   ```bash
   cd ml_lab
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pandas scikit-learn joblib flask numpy
   ```

## Usage

### 1. Train the Model

First, generate the training and test datasets, and train the model:

```bash
python train_model.py
```

This will:
- Load the original dataset
- Split data into 70% training and 30% testing
- Save split data to `data/train_data.csv` and `data/test_data.csv`
- Train a Linear Regression model
- Save the model as `temperature_lr_model.pkl`

### 2. Test the Model

Evaluate the trained model's performance:

```bash
python test_model.py
```

### 3. Run the Web Application

Start the Flask development server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

**Using the Application:**
1. Enter AQI value (Air Quality Index)
2. Enter Humidity percentage
3. Click "Predict" to get the predicted temperature
4. The result will display the predicted temperature and model confidence level

## Dependencies

- **pandas** - Data manipulation and CSV file handling
- **scikit-learn** - Machine learning algorithms and model evaluation
- **joblib** - Model serialization
- **flask** - Web framework
- **numpy** - Numerical computations

## Model Details

### Algorithm: Linear Regression

The model uses Linear Regression to establish a linear relationship between input features (AQI, Humidity) and the target variable (Temperature).

### Features
- **AQI (Air Quality Index):** Numerical value representing air pollution level
- **Humidity:** Percentage of humidity in the environment

### Target
- **Temperature:** Predicted temperature value in appropriate units

### Data Split
- Training Set: 70% of data
- Test Set: 30% of data
- Random State: 42 (for reproducibility)

## Notes

- The model confidence level is fixed at 70% (the R² score from training)
- Ensure the `temperature_lr_model.pkl` file exists before running the Flask app
- The Flask app runs in debug mode by default (suitable for development only)

## Future Enhancements

- Add more features (e.g., wind speed, precipitation)
- Implement more sophisticated models (Random Forest, Neural Networks)
- Add model retraining capabilities via the web interface
- Deploy to production server
- Add data validation and error handling

## Author

ML Lab Project

## License

MIT License
