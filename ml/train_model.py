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
- Save metrics as JSON
- Visualize evaluation using matplotlib
"""

import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# TRAIN MODEL
# =========================================================

def train(X_train, y_train, max_depth=10, criterion='gini'):

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, X_test, y_test, save_dir='results'):

    # Predict
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
    # Save JSON
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


# =========================================================
# PLOT METRICS GRAPH
# =========================================================

def plot_metrics(
    accuracy,
    precision,
    recall,
    f1,
    save_dir='results'
):

    metrics = [
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score'
    ]

    values = [
        accuracy,
        precision,
        recall,
        f1
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(metrics, values)

    plt.ylim(0, 1)

    plt.title('Decision Tree Evaluation Metrics')
    plt.ylabel('Score')

    # tampilkan value di atas bar
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f'{height:.4f}',
            ha='center'
        )

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f'{save_dir}/metrics_chart.png')

    plt.show()

    print(f"\nMetrics graph saved: {save_dir}/metrics_chart.png")


# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(cm, classes, save_dir='results'):

    cm_array = np.array(cm)

    plt.figure(figsize=(8, 6))

    plt.imshow(cm_array)

    plt.title('Confusion Matrix')

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # tampilkan angka di setiap cell
    for i in range(len(classes)):
        for j in range(len(classes)):

            plt.text(
                j,
                i,
                cm_array[i, j],
                ha='center',
                va='center'
            )

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f'{save_dir}/confusion_matrix.png')

    plt.show()

    print(f"\nConfusion matrix saved: {save_dir}/confusion_matrix.png")


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model, path='models/decision_tree.pkl'):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved: {path}")


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(path='models/decision_tree.pkl'):

    with open(path, 'rb') as f:
        return pickle.load(f)


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    # =========================
    # Load Dataset
    # =========================

    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')

    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    print("Training Decision Tree...")

    # =========================
    # Train Model
    # =========================

    model = train(
        X_train,
        y_train,
        max_depth=18,
        criterion='gini'
    )

    # =========================
    # Evaluate Model
    # =========================

    accuracy, precision, recall, f1, report, cm = evaluate(
        model,
        X_test,
        y_test
    )

    # =========================
    # Plot Metrics Graph
    # =========================

    plot_metrics(
        accuracy,
        precision,
        recall,
        f1
    )

    # =========================
    # Plot Confusion Matrix
    # =========================

    plot_confusion_matrix(
        cm,
        model.classes_
    )

    # =========================
    # Save Model
    # =========================

    save_model(model)