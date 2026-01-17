# Temperature Prediction

A machine learning application that predicts temperature based on Air Quality Index (AQI) and Humidity levels using Linear Regression.

## About

This project demonstrates a complete ML pipeline from data preprocessing, model training, and evaluation to a Flask-based web application for making predictions.

**Model Specifications:**
- **Algorithm:** Linear Regression
- **Features:** AQI, Humidity
- **Target:** Temperature
- **R² Score:** 0.70 (70% confidence level)

## Project Structure

```
temp-predictor-app/
├── app.py                              # Flask web application
├── train_model.py                      # Model training script
├── test_model.py                       # Model testing/evaluation
├── data/
│   ├── aqi_temp_humidity_normal_highcorr.csv   # Original dataset
│   ├── train_data.csv                  # 70% training data (generated)
│   └── test_data.csv                   # 30% testing data (generated)
├── templates/
│   ├── index.html                      # Input form page
│   └── result.html                     # Prediction results page
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd temp-predictor-app
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
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

### Training the Model

Generate training and test datasets, and train the model:

```bash
python train_model.py
```

This script will:
- Load the original dataset from `data/aqi_temp_humidity_normal_highcorr.csv`
- Split data into 70% training and 30% testing sets
- Save split data to `data/train_data.csv` and `data/test_data.csv`
- Train a Linear Regression model
- Save the trained model as `temperature_lr_model.pkl`

### Testing the Model

Evaluate the trained model's performance:

```bash
python test_model.py
```

### Running the Web Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

**Using the Application:**
1. Enter the AQI (Air Quality Index) value
2. Enter the Humidity percentage
3. Click "Predict" to generate a temperature prediction
4. View the predicted temperature and model confidence level on the results page

## Dependencies

- **pandas** - Data manipulation and CSV file handling
- **scikit-learn** - Machine learning algorithms and model evaluation
- **joblib** - Model serialization and persistence
- **flask** - Web framework for the application
- **numpy** - Numerical computations

## Model Information

### Algorithm

Linear Regression is used to establish a linear relationship between input features and the target temperature variable.

### Features

- **AQI (Air Quality Index):** Numerical value representing air pollution level
- **Humidity:** Percentage of atmospheric humidity

### Target Variable

- **Temperature:** Predicted environmental temperature

### Data Specifications

- **Training Set:** 70% of data (random split)
- **Test Set:** 30% of data
- **Random State:** 42 (ensures reproducibility)

## Important Notes

- The model must be trained before running the Flask application (requires `temperature_lr_model.pkl`)
- The Flask app runs in debug mode by default (development only)
- Model confidence is fixed at 70% (R² score from training)
- Ensure the `requirements.txt` file contains all dependencies

## Future Enhancements

- Expand features (wind speed, precipitation, pressure, etc.)
- Implement advanced models (Random Forest, Neural Networks, XGBoost)
- Add real-time model retraining via web interface
- Production deployment with WSGI server
- Enhanced data validation and error handling
- Unit tests and integration tests
- API documentation with Swagger/OpenAPI
- Model versioning and tracking
