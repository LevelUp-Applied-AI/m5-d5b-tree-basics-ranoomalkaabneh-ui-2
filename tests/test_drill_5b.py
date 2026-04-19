"""Autograder tests for Drill 5B — Tree-Based Model Basics."""

import ast
import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drill import train_decision_tree, get_feature_importances, train_balanced_forest

FEATURES = ["tenure", "monthly_charges", "total_charges",
            "num_support_calls", "senior_citizen", "has_partner",
            "has_dependents", "contract_months"]


@pytest.fixture
def data():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv")
    )
    X = df[FEATURES]
    y = df["churned"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def test_decision_tree_trained(data):
    X_train, X_test, y_train, y_test = data
    model = train_decision_tree(X_train, y_train)
    assert model is not None, "train_decision_tree returned None"
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(model, DecisionTreeClassifier), "Must return DecisionTreeClassifier"
    assert model.max_depth == 5, f"max_depth should be 5, got {model.max_depth}"
    assert hasattr(model, "classes_"), "Model must be fitted"


def test_feature_importances(data):
    X_train, X_test, y_train, y_test = data
    model = train_decision_tree(X_train, y_train)
    assert model is not None
    importances = get_feature_importances(model, FEATURES)
    assert importances is not None, "get_feature_importances returned None"
    assert len(importances) == len(FEATURES), f"Expected {len(FEATURES)} features"
    total = sum(importances.values())
    assert abs(total - 1.0) < 0.01, f"Importances should sum to ~1.0, got {total}"
    values = list(importances.values())
    assert values == sorted(values, reverse=True), "Importances should be sorted descending"


def test_random_forest_balanced_returns_valid_metrics(data):
    """Verify train_balanced_forest returns a valid metrics dict.

    On this dataset at the default 0.5 threshold, the balanced
    RandomForestClassifier legitimately produces very low or zero precision,
    recall, and F1 because class-1 predicted probabilities rarely cross 0.5.
    This is a real property of imbalanced data with tree models — covered in
    the Week B reading §4-6. The autograder validates that metrics are
    correctly-typed floats in [0, 1] rather than requiring strictly positive
    values.
    """
    X_train, X_test, y_train, y_test = data
    metrics = train_balanced_forest(X_train, y_train, X_test, y_test)
    assert metrics is not None, "train_balanced_forest returned None"
    for key in ["precision", "recall", "f1"]:
        assert key in metrics, f"Missing key: {key}"
        value = metrics[key]
        assert isinstance(value, (int, float, np.integer, np.floating)), (
            f"{key} must be a number, got {type(value).__name__}"
        )
        assert 0.0 <= float(value) <= 1.0, (
            f"{key} should be in [0, 1], got {value}"
        )


def test_balanced_forest_implementation_is_real():
    """AST check: verify train_balanced_forest actually trains a balanced RF and computes metrics.

    Prevents hardcoded-dict stubs from silently passing. A correct
    implementation must call RandomForestClassifier, .fit(), and the three
    scikit-learn metric functions, and must use the literal 'balanced' for
    class_weight.
    """
    source = inspect.getsource(train_balanced_forest)
    tree = ast.parse(source)

    call_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                call_names.add(node.func.id)

    assert "RandomForestClassifier" in call_names, (
        "train_balanced_forest must call RandomForestClassifier"
    )
    assert "fit" in call_names, "train_balanced_forest must call .fit()"
    for metric_fn in ("precision_score", "recall_score", "f1_score"):
        assert metric_fn in call_names, (
            f"train_balanced_forest must call {metric_fn}"
        )
    assert "'balanced'" in source or '"balanced"' in source, (
        "train_balanced_forest must use class_weight='balanced'"
    )
