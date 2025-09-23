# Diabetes Prediction App

This application predicts the likelihood of diabetes based on patient health metrics using machine learning.

## Features

- Data exploration and filtering
- Interactive visualizations
- Diabetes prediction with confidence scores
- Model performance evaluation

## Dataset

The Pima Indians Diabetes Dataset contains health metrics from 768 patients with 8 features:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age
9. Outcome (0 = No Diabetes, 1 = Diabetes)

## Installation

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Model Training

The model was trained using a Random Forest classifier with 100 trees. The dataset was split into 80% training and 20% testing. Feature scaling was applied using StandardScaler.

## Deployment

The app can be deployed on Streamlit Cloud by connecting to this GitHub repository.

## Files

- `app.py`: Streamlit application
- `model.pkl`: Trained model
- `scaler.pkl`: Feature scaler
- `diabetes.csv`: Dataset
- `requirements.txt`: Python dependencies
- `README.md`: This file