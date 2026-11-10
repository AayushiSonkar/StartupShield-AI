import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(name, model, x_test, y_test):
    """Return standard binary-classification metrics for one trained model."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


def comparison_table(results):
    """Create an F1-score-sorted comparison table."""
    return pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
