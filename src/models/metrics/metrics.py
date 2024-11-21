import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(binary_preds, labels):
    # Ensure the inputs are numpy arrays
    binary_preds = np.array(binary_preds)
    labels = np.array(labels)

    # Calculate metrics
    metrics = {
        'precision': precision_score(labels, binary_preds, average='binary', zero_division=0),
        'recall': recall_score(labels, binary_preds, average='binary', zero_division=0),
        'f1_score': f1_score(labels, binary_preds, average='binary', zero_division=0),
        'accuracy': accuracy_score(labels, binary_preds)
    }
    return metrics
