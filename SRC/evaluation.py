"""
Purpose:
    Evaluates the trained models and generates visual performance insights.
Key responsibilities:
    Predict results using the test set
    Calculate performance metrics:
        Accuracy
        Precision
        Recall
        F1-Score
        ROC-AUC
    Plot confusion matrices
    Plot ROC curves
    Create comparison tables for models
Typical functions inside:
    evaluate_model(name, y_true, y_pred)
    plot_confusion_matrix(model_name, y_true, y_pred)
    plot_roc_curve(models, X_test, y_test)
    compare_accuracies(models, y_test, predictions)
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model(name, y_test, y_pred):
    """
    Prints accuracy, precision, recall, F1 score, and ROC-AUC
    """

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))

def compare_accuracies(model_names, predictions, y_test):
    """
    Builds a table of accuracies for all models
    """

    accuracies = [accuracy_score(y_test, pred) for pred in predictions]

    results = pd.DataFrame({
        "Model": model_names,
        "Accuracy": accuracies
    })

    print("\nModel Accuracy Comparison:\n")
    print(results)


def plot_confusion_matrices(models_dict, y_test):
    """
    Plots confusion matrices of all models
    """

    plt.figure(figsize=(14, 10))

    i = 1
    for name, pred in models_dict.items():
        plt.subplot(2, 2, i)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        i += 1

    plt.tight_layout()
    plt.show()
