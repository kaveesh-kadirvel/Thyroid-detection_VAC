# 🩺 Thyroid Disease Prediction System

A Machine Learning based web application that predicts the risk of thyroid disease recurrence using clinical and patient-related parameters.

## Features

- User-friendly Streamlit interface
- Real-time prediction
- Data preprocessing using Label Encoding and Standard Scaling
- Trained Machine Learning model
- Confidence score for predictions

## Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy

## Dataset Features

- Age
- Gender
- Smoking
- Hx Smoking
- Hx Radiotherapy
- Thyroid Function
- Physical Examination
- Adenopathy
- Pathology
- Focality
- Risk
- TNM Staging
- Response

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/thyroid-disease-prediction.git
cd thyroid-disease-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Workflow

1. Load trained model
2. Load training dataset
3. Encode categorical variables
4. Scale features
5. Take user input
6. Predict recurrence risk
7. Display result with confidence score

## Future Improvements

- Deploy on Streamlit Cloud
- Add SHAP Explainability
- Improve UI/UX
- Add Patient History Dashboard
- Support Multiple Thyroid Conditions

## Disclaimer

This project is developed for educational and research purposes only and should not be considered a substitute for professional medical advice.
