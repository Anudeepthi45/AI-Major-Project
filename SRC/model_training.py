"""
Purpose:
    Responsible for training all machine-learning models used in the project.
Key responsibilities:
  Initialize models like:
      Logistic Regression
      Decision Tree
      Random Forest
      Neural Network (MLPClassifier)
  Train each model on training data
  Return trained models for evaluation
  Save best-performing model (using joblib)
Typical functions inside:
    train_logistic_regression(X_train, y_train)
    train_decision_tree(X_train, y_train)
    train_random_forest(X_train, y_train)
    train_neural_network(X_train, y_train)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_models(X_train, y_train):
    """
    Trains LR, DT, RF, NN models and returns them
    """

    lr = LogisticRegression(max_iter=1000)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=2000, random_state=42)

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    nn.fit(X_train, y_train)

    return lr, dt, rf, nn
