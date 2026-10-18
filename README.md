
# Gastrointestinal Disease Classification Using Machine Learning

## Overview
This project demonstrates gastrointestinal (GI) disease-risk classification using simulated patient data and four machine learning algorithms.

## Machine Learning Models
- LightGBM
- XGBoost
- Random Forest
- Logistic Regression

## Dataset Features
The simulated dataset includes:
- Age and BMI
- Fecal calprotectin
- Microbiome index
- H. pylori status
- Diet type
- Food intolerance
- Stress level
- NSAID use

## Methodology
1. Generate simulated patient data.
2. Preprocess categorical features using one-hot encoding.
3. Split the dataset into training and testing sets.
4. Train four machine learning classifiers.
5. Evaluate performance using accuracy, ROC-AUC, precision, recall, and F1-score.
6. Interpret model predictions using SHAP summary plots.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- LightGBM
- XGBoost
- SHAP

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python gi_disease_classifier.py
```

## Important Note
This project uses simulated data for educational and research prototyping purposes. It is not a clinically validated diagnostic system and must not be used to diagnose real patients.
