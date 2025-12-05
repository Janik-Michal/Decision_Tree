import numpy as np
import pytest
from decision_tree.decision_tree import DecisionTreeClassifier, entropy, gini, SKTree
from decision_tree.decision_tree import compare


# ------------------------
# Unit tests for utilities
# ------------------------

def test_entropy_single_label():
    y = np.array([0, 0, 0])
    assert entropy(y) == 0.0


def test_gini_single_label():
    y = np.array([1, 1, 1, 1])
    assert gini(y) == 0.0


def test_entropy_multiple_labels():
    y = np.array([0, 0, 1, 1])
    ent = entropy(y)
    assert np.isclose(ent, 1.0)


def test_gini_multiple_labels():
    y = np.array([0, 0, 1, 1])
    g = gini(y)
    assert np.isclose(g, 0.5)


# ------------------------
# Unit tests for DecisionTreeClassifier
# ------------------------

def test_fit_predict_and_logic():
    # Simple binary dataset
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)

    assert np.array_equal(preds, y)


def test_score_accuracy():
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    acc = clf.score(X, y)
    assert acc == 1.0


# ---------------------------------------------
# Integration test — only runs if sklearn exists
# ---------------------------------------------

@pytest.mark.skipif(SKTree is None, reason="scikit-learn not installed")
def test_compare_runs_without_error(capfd):
    """
    Just check that compare() runs and prints output.
    """
    compare(max_depth=2)

    captured = capfd.readouterr()
    assert "Comparison" in captured.out
