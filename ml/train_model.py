"""
Checkpoint 4: Model Training
- Train Decision Tree Classifier
- Hyperparameter tuning
- Save model as .pkl
"""

import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import json


def train(X_train, y_train, max_depth=10, criterion='gini'):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, save_dir='results'):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_).tolist()

    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump({'accuracy': acc, 'report': report, 'confusion_matrix': cm,
                   'classes': list(model.classes_)}, f, indent=2)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc, report, cm


def save_model(model, path='models/decision_tree.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")


def load_model(path='models/decision_tree.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    print("Training Decision Tree...")
    model = train(X_train, y_train, max_depth=10)
    evaluate(model, X_test, y_test)
    save_model(model)