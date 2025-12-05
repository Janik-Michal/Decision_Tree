import numpy as np
import pytest
from decision_tree.decision_tree import DecisionTreeClassifier, _entropy, _gini, compare_with_sklearn

# ------------------------
# Unit tests for utilities
# ------------------------

def test_entropy_single_label():
    y = np.array([0, 0, 0])
    assert _entropy(y) == 0.0

def test_gini_single_label():
    y = np.array([1, 1, 1, 1])
    assert _gini(y) == 0.0

def test_entropy_multiple_labels():
    y = np.array([0, 0, 1, 1])
    ent = _entropy(y)
    assert np.isclose(ent, 1.0)

def test_gini_multiple_labels():
    y = np.array([0, 0, 1, 1])
    g = _gini(y)
    assert np.isclose(g, 0.5)

# ------------------------
# Unit tests for DecisionTreeClassifier
# ------------------------

def test_fit_predict_and_logic():
    # AND-like dataset z 4 próbkami na klasę (dla min_samples_split=2)
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2)  # domyślny min_samples_split=2
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)

def test_predict_proba_one_hot():
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    # probabilities should be one-hot
    assert probs.shape == (4, 2)
    assert np.all(np.sum(probs, axis=1) == 1.0)
    assert np.all((probs == 0) | (probs == 1))

def test_score_accuracy():
    # AND-like dataset dla poprawnej klasyfikacji
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc == 1.0

# ------------------------
# Integration / comparison test
# ------------------------

@pytest.mark.skipif(compare_with_sklearn is None, reason="scikit-learn not installed")
def test_compare_with_sklearn_runs():
    report = compare_with_sklearn(random_state=42, test_size=0.3, max_depth=3)
    assert "my_accuracy" in report
    assert "sklearn_accuracy" in report
    # accuracies should be floats
    assert isinstance(report["my_accuracy"], float)
    assert isinstance(report["sklearn_accuracy"], float)
    # predictions should be lists
    assert isinstance(report["my_pred"], list)
    assert isinstance(report["sklearn_pred"], list)
