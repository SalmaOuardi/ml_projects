import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)


def compute_all_metrics(y_true, y_pred, y_probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "pr_auc": average_precision_score(y_true, y_probs),
        "specificity": specificity
    }

def save_plots(y_true, y_pred, y_probs, model_name):
    os.makedirs("results/plots", exist_ok=True)

    # Confusion Matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.savefig(f"results/plots/confusion_matrix_{model_name}.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.savefig(f"results/plots/roc_curve_{model_name}.png")

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.savefig(f"results/plots/pr_curve_{model_name}.png")
    
    
def save_metrics(metrics_dict, model_name):
    path = "results/metrics.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_name] = metrics_dict

    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
def evaluate_model(model, X_test, y_test, model_name):
    if model_name == "isolation":
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]
        y_probs = y_pred  # Not probabilistic
    else:
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_all_metrics(y_test, y_pred, y_probs)
    save_metrics(metrics, model_name)
    save_plots(y_test, y_pred, y_probs, model_name)

    print(f"📊 Metrics for {model_name}")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")