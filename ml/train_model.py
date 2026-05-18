"""
Checkpoint 4: Model Training
- Train Decision Tree Classifier
- Evaluation Metrics:
    Accuracy
    Precision
    Recall
    F1-score
    Confusion Matrix
- Save model as .pkl
"""

import numpy as np
import pickle
import os
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def train(X_train, y_train, max_depth=10, criterion='gini'):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, save_dir='results'):

    # Prediksi
    y_pred = model.predict(X_test)

    # =========================
    # Metrics
    # =========================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    ).tolist()

    # =========================
    # Save Results
    # =========================

    os.makedirs(save_dir, exist_ok=True)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'classes': list(model.classes_)
    }

    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # =========================
    # Print Results
    # =========================

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    print("\n===== CONFUSION MATRIX =====")
    print(np.array(cm))

    return accuracy, precision, recall, f1, report, cm


def save_model(model, path='models/decision_tree.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved: {path}")


def load_model(path='models/decision_tree.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    # Load dataset
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')

    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    print("Training Decision Tree...")

    # Train model
    model = train(
        X_train,
        y_train,
        max_depth=18,
        criterion='gini'
    )

    # Evaluate
    evaluate(model, X_test, y_test)

    # Save model
    save_model(model)