"""Full-batch gradient descent. Pass numeric, prepared features."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from grader_contracts.linear_models import (
    BaselineComparisonResult, ClassificationInput, ClassificationResult,
    ClassificationMetrics, RegressionInput, RegressionResult,
)


def _arrays(data):
    x = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float).reshape(-1)
    test = np.asarray(data.test_features, dtype=float)
    if x.ndim != 2 or test.ndim != 2 or x.shape[1] != test.shape[1]:
        raise ValueError('Features must be 2D with matching columns.')
    if len(x) == 0 or len(x) != len(y):
        raise ValueError('Training features and targets must have equal nonzero length.')
    if not all(np.isfinite(a).all() for a in (x, y, test)):
        raise ValueError('Inputs must contain finite numbers.')
    if data.learning_rate <= 0 or data.epochs < 1:
        raise ValueError('learning_rate and epochs must be positive.')
    return x, y, test


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Minimize MSE with full-batch gradient descent; return post-update losses."""
    x, y, test = _arrays(data)
    weights, bias, history = np.zeros(x.shape[1]), 0.0, []
    for _ in range(data.epochs):
        error = x @ weights + bias - y
        weights -= data.learning_rate * (2.0 / len(y)) * (x.T @ error)
        bias -= data.learning_rate * 2.0 * error.mean()
        history.append(float(np.mean((x @ weights + bias - y) ** 2)))
    return RegressionResult(weights, float(bias), history, test @ weights + bias)


def _sigmoid(z):
    # Stable for both very large positive and negative scores.
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Minimize binary cross-entropy using full-batch gradient descent."""
    x, y, test = _arrays(data)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Classification targets must be 0 or 1.')
    weights, bias, history = np.zeros(x.shape[1]), 0.0, []
    for _ in range(data.epochs):
        error = _sigmoid(x @ weights + bias) - y
        weights -= data.learning_rate * (x.T @ error) / len(y)
        bias -= data.learning_rate * error.mean()
        scores = x @ weights + bias
        history.append(float(np.mean(np.logaddexp(0, scores) - y * scores)))
    probabilities = _sigmoid(test @ weights + bias)
    return ClassificationResult(weights, float(bias), history,
                                probabilities, (probabilities >= 0.5).astype(int))


def classification_metrics(y, probabilities, predictions):
    return ClassificationMetrics(
        float(roc_auc_score(y, probabilities)), float(accuracy_score(y, predictions)),
        float(precision_score(y, predictions, zero_division=0)),
        float(recall_score(y, predictions, zero_division=0)),
        float(f1_score(y, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Return the three sklearn baselines required by the fixed contract.

    run_experiments.py saves own_logistic and all three baselines in JSON.
    """
    x, y, test = _arrays(data)
    results = []
    for model in (LogisticRegression(max_iter=2000, random_state=42),
                  DecisionTreeClassifier(random_state=42),
                  KNeighborsClassifier(n_neighbors=5)):
        model.fit(x, y)
        positive_column = list(model.classes_).index(1)
        results.append(classification_metrics(data.test_target,
            model.predict_proba(test)[:, positive_column], model.predict(test)))
    return BaselineComparisonResult(*results)
