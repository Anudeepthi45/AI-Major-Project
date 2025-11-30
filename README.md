## Heart Disease Prediction using Machine Learning Classification Models

## Project Overview

This project focuses on predicting whether a patient is likely to have heart disease based on medical attributes such as age, sex, cholesterol levels, chest pain type, blood pressure, ECG results, maximum heart rate, and more.
Using machine learning classification models, the system predicts the target variable:
    1 → Presence of heart disease
    0 → No heart disease
This predictive approach helps hospitals, health organizations, and doctors identify high-risk patients earlier, enabling preventive care and better clinical decisions.

## Objectives
-> To analyze heart disease data and understand key health indicators.
-> To clean, preprocess, and scale the dataset for ML tasks.
-> To build multiple machine learning models:
      >> Logistic Regression
      >> Decision Tree
      >> Random Forest
      >> Neural Network (MLP)
-> To evaluate each model using:
      >> Accuracy
      >> Precision
      >> Recall
      >> F1-score
      >> ROC-AUC
      >> Confusion Matrix
-> To identify the most influential features contributing to heart disease prediction.

## Dataset Description
The dataset contains patient medical attributes such as:
    --> age – Patient age
    --> sex – 1 = male, 0 = female
    --> cp – Chest pain type
    --> trestbps – Resting blood pressure
    --> chol – Serum cholesterol
    --> fbs – Fasting blood sugar
    --> restecg – Resting ECG results
    --> thalach – Maximum heart rate
    --> exang – Exercise-induced angina
    --> oldpeak – ST depression
    --> slope – Slope of peak exercise
    --> ca – Number of major vessels
    --> thal – Thalassemia
    --> target – 1 = Heart disease, 0 = No heart disease
Dataset used: heart.csv
The dataset was checked for missing values and cleaned accordingly. Numerical features were scaled using StandardScaler.

## Methodology
1. Data Preprocessing
-> Loaded the dataset using pandas
-> Replaced missing values with mean
-> Separated features (X) and target (y)
-> Scaled the numeric features using StandardScaler
-> Split data into 80% training and 20% testing

2. Model Training
-> Trained four ML models:
    >> Logistic Regression
    >> Decision Tree Classification
    >> Random Forest Classification
    >> Neural Network (MLPClassifier)

3. Predictions
-> Each model predicted heart disease outcomes on the test dataset.

4. Model Evaluation
-> Evaluated the models using:
        >> Accuracy
        >> Precision
        >> Recall
        >> F1 Score
        >> ROC-AUC
        >> Classification Report
        >> Confusion Matrix plots for all models

## Results
* The models achieved the following performance:
* Model -- Accuracy
    >> Logistic Regression   -- 0.7869
    >> Decision Tree         -- 0.7049
    >> Random Forest         -- 0.8196
    >> Neural Network (MLP)  -- 0.7869
Key Findings
  -> Random Forest achieved the highest accuracy (81.96%), making it the best-performing model.
  -> Logistic Regression and Neural Network tied with 78.69% accuracy.
  -> Decision Tree showed relatively lower performance due to overfitting.
  -> Important features influencing prediction include:
      * chest pain type (cp)
      * maximum heart rate (thalach)
      * oldpeak
      * number of major vessels (ca)
      * ST slope (slope)
      
## Tools and Technologies
  * Python
  * Libraries:
      >> pandas, numpy
      >> seaborn, matplotlib
      >> scikit-learn

## Key Outcomes
-> Successfully developed and evaluated multiple ML models for heart disease prediction.
-> Identified feature importance and gained insights into cardiovascular risk indicators.
-> Random Forest model exported as random_forest_model.pkl for future deployment.
-> Provides a foundation for medical decision-support and early risk analysis systems.

## Author
Name: Thotakura Anudeepthi 
Project: Artificial Intelligence Major Project 
Institution: KL UNIVERSITY
