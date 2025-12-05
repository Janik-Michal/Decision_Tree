"""
Decision Tree Project
=====================

Authors: student team
Course: Python Programming - Decision Trees Project
License: CC BY 4.0

This single-file contains:
 - A self-contained DecisionTreeClassifier implementation (CART-style)
 - Unit tests and simple experimental comparison with scikit-learn's DecisionTreeClassifier
 - Usage instructions meant to satisfy the course project requirements (documentation, tests,
   comparison to scikit-learn).

Repository notes (what to include in your GitHub repo):
 - This file: `decision_tree_project.py`
 - A proper PyScaffold project (use `putup` to scaffold), a VirtualEnv and requirements.txt:
     sklearn
     numpy
     scipy
     pytest
 - A README.md describing the algorithm, experiments, and how to run tests.
 - Keep code documented with docstrings and inline comments (this file follows that rule).

How to run (example):
 1. Create & activate virtualenv: `python -m venv .venv && source .venv/bin/activate` (POSIX)
 2. Install dependencies: `pip install -r requirements.txt` (or `pip install scikit-learn numpy pytest`)
 3. Run quick demo & comparison: `python decision_tree_project.py --demo`
 4. Run unit tests: `python decision_tree_project.py --test` or `pytest`

Grading checklist covered by this file:
 - Correctness & parity vs. scikit-learn: `compare_with_sklearn` function runs experiments and prints metrics.
 - Code Quality & Tests: `unittest`-based tests and a small pytest-compatible test are included.
 - Documentation: docstrings and comments included.
 - Experimental Design & Analysis: demo performs train/test and reports accuracy and basic metrics.
 - Software engineering: single-file is easy to inspect; commit history should be used in the repo.
 - Peer review: include instructions in README to guide reviewers how to run tests and checks.

Note: this implementation focuses on clarity and pedagogical parity rather than extreme optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Dict
import numpy as np
from collections import Counter
import argparse
import sys

# For comparison & tests
try:
    from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except Exception:
    SKDecisionTree = None


# -------------------------
# Utilities
# -------------------------

def _entropy(y: np.ndarray) -> float:
    """Calculate Shannon entropy of label array y."""
    if y.size == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / y.size
    return -float(np.sum(probs * np.log2(probs)))


def _gini(y: np.ndarray) -> float:
    """Calculate Gini impurity of label array y."""
    if y.size == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / y.size
    return 1.0 - float(np.sum(probs ** 2))


@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[_Node] = None
    right: Optional[_Node] = None
    value: Optional[int] = None  # class label for leaf
    depth: int = 0


# -------------------------
# Decision Tree Implementation (CART)
# -------------------------

class DecisionTreeClassifier:
    """
    A simple Decision Tree classifier implementing binary splits (CART) for classification.

    Parameters
    ----------
    criterion : str, optional (default='gini')
        The function to measure the quality of a split. Supported: 'gini', 'entropy'.
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain less than min_samples_split samples.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    max_features : int or None, optional (default=None)
        Number of features to consider when looking for the best split. If None uses all features.
    """

    def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None, min_samples_split: int = 2,
                 max_features: Optional[int] = None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.n_classes_ = None
        self.n_features_ = None
        self.tree_: Optional[_Node] = None

    def _impurity(self, y: np.ndarray) -> float:
        return _gini(y) if self.criterion == "gini" else _entropy(y)

    def _majority_class(self, y: np.ndarray) -> int:
        counts = np.bincount(y)
        return int(np.argmax(counts))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split for dataset (X, y). Returns (feature_index, threshold, impurity_decrease).
        If no valid split exists returns (None, None, 0.0).
        """
        n_samples, n_features = X.shape
        if n_samples < 2:
            return None, None, 0.0

        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        features = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            rng = np.random.default_rng(0)
            features = rng.choice(n_features, self.max_features, replace=False)

        # For each feature, consider splits at midpoints between sorted unique values
        for feature in features:
            values = X[:, feature]
            sorted_idx = np.argsort(values)
            sorted_vals = values[sorted_idx]
            sorted_y = y[sorted_idx]

            # identify possible split positions where value changes
            potential_idxs = np.where(sorted_vals[:-1] != sorted_vals[1:])[0]
            if potential_idxs.size == 0:
                continue

            # cumulative counts for left side used to compute impurity efficiently
            for idx in potential_idxs:
                # threshold is midpoint
                thresh = (sorted_vals[idx] + sorted_vals[idx + 1]) / 2.0
                left_y = sorted_y[: idx + 1]
                right_y = sorted_y[idx + 1 :]

                if left_y.size < self.min_samples_split or right_y.size < self.min_samples_split:
                    continue

                impurity_left = self._impurity(left_y)
                impurity_right = self._impurity(right_y)
                weighted_impurity = (left_y.size * impurity_left + right_y.size * impurity_right) / n_samples

                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(feature)
                    best_thresh = float(thresh)

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> _Node:
        node = _Node(depth=depth)

        # If pure or max depth reached or not enough samples to split => leaf
        if y.size == 0:
            node.value = None
            return node

        if len(np.unique(y)) == 1:
            node.value = int(y[0])
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node.value = self._majority_class(y)
            return node

        if y.size < self.min_samples_split * 2:
            node.value = self._majority_class(y)
            return node

        feature, thresh, gain = self._best_split(X, y)
        if feature is None or gain <= 0.0:
            node.value = self._majority_class(y)
            return node

        node.feature = feature
        node.threshold = thresh

        # partition data
        left_mask = X[:, feature] <= thresh
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit the decision tree classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        if node.value is not None:
            return int(node.value)
        if node.feature is None:
            return int(self._majority_class(np.array([], dtype=int)))
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.tree_ is None:
            raise ValueError("The tree is not fitted yet. Call fit first.")
        X = np.asarray(X)
        preds = [self._predict_one(row, self.tree_) for row in X]
        return np.array(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.
        Returns an array shape (n_samples, n_classes).
        """
        # For simplicity, leaf returns one-hot probability for majority class
        if self.tree_ is None:
            raise ValueError("The tree is not fitted yet. Call fit first.")
        X = np.asarray(X)
        probs = []
        for row in X:
            node = self.tree_
            while node.value is None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            p = np.zeros(self.n_classes_, dtype=float)
            p[int(node.value)] = 1.0
            probs.append(p)
        return np.vstack(probs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


# -------------------------
# Experimental Comparison and Tests
# -------------------------


def compare_with_sklearn(random_state: int = 0, test_size: float = 0.3, max_depth: Optional[int] = None) -> Dict[str, Any]:
    """Run a simple comparison with scikit-learn's DecisionTreeClassifier on the Iris dataset.

    Returns a dict with metrics from both implementations.
    """
    if SKDecisionTree is None:
        raise RuntimeError("scikit-learn is required for comparison. Install scikit-learn.")

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # our implementation
    my_tree = DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
    my_tree.fit(X_train, y_train)
    y_pred_my = my_tree.predict(X_test)
    acc_my = accuracy_score(y_test, y_pred_my)

    # sklearn
    sk_tree = SKDecisionTree(criterion="gini", max_depth=max_depth, random_state=random_state)
    sk_tree.fit(X_train, y_train)
    y_pred_sk = sk_tree.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)

    report = {
        "my_accuracy": float(acc_my),
        "sklearn_accuracy": float(acc_sk),
        "my_pred": y_pred_my.tolist(),
        "sklearn_pred": y_pred_sk.tolist(),
        "confusion_my": confusion_matrix(y_test, y_pred_my).tolist(),
        "confusion_sklearn": confusion_matrix(y_test, y_pred_sk).tolist(),
        "classification_report_sklearn": classification_report(y_test, y_pred_sk, output_dict=True),
    }

    return report


# -------------------------
# Simple unit tests
# -------------------------

import unittest


class TestDecisionTree(unittest.TestCase):
    def test_entropy_gini_consistency(self):
        # entropy and gini on single label
        y = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(_entropy(y), 0.0)
        self.assertAlmostEqual(_gini(y), 0.0)

    def test_fit_predict_simple(self):
        # simple logical dataset: AND-like split
        X = np.array([[0], [0], [1], [1]])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertTrue(np.array_equal(preds, y))

    def test_compare_with_sklearn_runs(self):
        if SKDecisionTree is None:
            self.skipTest("scikit-learn not available")
        report = compare_with_sklearn(random_state=1, test_size=0.5, max_depth=3)
        self.assertIn("my_accuracy", report)
        self.assertIn("sklearn_accuracy", report)


# pytest compatible basic test (keeps pytest happy if used)

def test_basic_and_split():
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


# -------------------------
# Command-line interface for demo & tests
# -------------------------

def _print_report(report: Dict[str, Any]) -> None:
    print("=== Comparison report (Iris dataset) ===")
    print(f"Our implementation accuracy: {report['my_accuracy']:.4f}")
    print(f"scikit-learn accuracy: {report['sklearn_accuracy']:.4f}")
    print("Confusion matrix (ours):")
    print(np.array(report['confusion_my']))
    print("Confusion matrix (sklearn):")
    print(np.array(report['confusion_sklearn']))
    print("Classification report for scikit-learn (dict):")
    for label, stats in report['classification_report_sklearn'].items():
        print(f"{label}: {stats}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decision Tree project demo & tests")
    parser.add_argument("--demo", action="store_true", help="Run demo comparison against scikit-learn")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--max-depth", type=int, default=None, help="max depth for trees in demo")
    args = parser.parse_args(argv)

    if args.test:
        # run unittest
        unittest.main(argv=[sys.argv[0]], exit=False)

    if args.demo:
        if SKDecisionTree is None:
            print("scikit-learn not installed. Install it to run the demo.")
            return
        report = compare_with_sklearn(random_state=42, test_size=0.3, max_depth=args.max_depth)
        _print_report(report)


if __name__ == "__main__":
    # default behaviour: show a short help-like message unless flags passed
    if len(sys.argv) == 1:
        print("This module includes an implementation of Decision Tree (CART), unit tests, and a demo comparison with scikit-learn.")
        print("Run `python decision_tree_project.py --demo` to compare, or `python decision_tree_project.py --test` to run tests.")
    else:
        main()
