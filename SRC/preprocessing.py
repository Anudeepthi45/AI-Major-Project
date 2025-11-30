"""
Purpose:
    Handles all data loading, cleaning, preprocessing, and train-test splitting steps.
    This makes the dataset ready for machine learning models.
Key responsibilities:
    Load the CSV dataset
    Handle missing values
    Separate input features (X) and target (y)
    Perform train–test split
    Apply feature scaling using StandardScaler
    Return X_train, X_test, y_train, y_test
Typical functions inside:
  load_data(path)
  clean_data(df)
  split_data(df)
  scale_data(X_train, X_test)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    """
    Loads the heart disease dataset
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    Handles missing values and splits data
    """
    # Fill missing values (if any)
    df.fillna(df.mean(), inplace=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
