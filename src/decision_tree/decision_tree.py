"""
Simple Decision Tree (CART) implementation
Author: Michał Janik
"""

import numpy as np
from dataclasses import dataclass
import argparse
import sys

# Optional sklearn import for comparison demo
try:
    from sklearn.tree import DecisionTreeClassifier as SKTree
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except:
    SKTree = None


# -----------------------------
# Basic impurity metrics
# -----------------------------

def entropy(y):
    """Compute Shannon entropy for a label array."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))


def gini(y):
    """Compute Gini impurity for a label array."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return 1 - np.sum(probs ** 2)


# Node structure for the tree
@dataclass
class Node:
    feature: int = None
    threshold: float = None
    left: "Node" = None
    right: "Node" = None
    value: int = None  # class label for leaf nodes


# -----------------------------
# Main Decision Tree class
# -----------------------------

class DecisionTreeClassifier:
    """
    A simple Decision Tree classifier using binary splits (CART).
    Designed to be easy to understand (student-friendly).
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    # Select impurity metric
    def _impurity(self, y):
        """Return impurity value using selected criterion."""
        return gini(y) if self.criterion == "gini" else entropy(y)

    def _majority(self, y):
        """Return the class that appears most often."""
        return np.bincount(y).argmax()

    # Core split search
    def _best_split(self, X, y):
        """
        Find the best feature and threshold that give the highest impurity reduction.
        Returns (feature, threshold, gain).
        """
        n, m = X.shape
        best_feat = None
        best_thresh = None
        best_gain = 0
        parent_impurity = self._impurity(y)

        # Try splitting on each feature
        for f in range(m):
            values = X[:, f]
            sorted_idx = np.argsort(values)
            values = values[sorted_idx]
            labels = y[sorted_idx]

            # Check all possible split points
            for i in range(1, n):
                if values[i] == values[i - 1]:
                    continue  # no new split point

                thresh = (values[i] + values[i - 1]) / 2
                left = labels[:i]
                right = labels[i:]

                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue

                # Weighted impurity
                impurity = (
                    len(left) * self._impurity(left) +
                    len(right) * self._impurity(right)
                ) / n

                gain = parent_impurity - impurity

                # Keep the best split
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    # Recursively build the tree
    def _build(self, X, y, depth=0):
        """Recursively build the decision tree and return root node."""
        # Stopping condition: pure node
        if len(set(y)) == 1:
            return Node(value=y[0])

        # Stop if max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._majority(y))

        # Try to find a good split
        feat, thresh, gain = self._best_split(X, y)

        # If no good split — create a leaf
        if feat is None or gain == 0:
            return Node(value=self._majority(y))

        # Split data
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        """Train the decision tree classifier."""
        X = np.array(X)
        y = np.array(y, dtype=int)
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """Traverse the tree to predict a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """Predict labels for multiple samples."""
        X = np.array(X)
        return np.array([self._predict_one(row, self.tree_) for row in X])

    def score(self, X, y):
        """Compute accuracy of the model."""
        pred = self.predict(X)
        return (pred == y).mean()


# -----------------------------
# Demo: compare with sklearn
# -----------------------------

def compare(max_depth=None):
    """Train both this tree and sklearn's tree on the Iris dataset and print accuracies."""
    if SKTree is None:
        print("scikit-learn not installed – demo disabled.")
        return

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Our tree
    my = DecisionTreeClassifier(max_depth=max_depth)
    my.fit(X_train, y_train)
    acc_my = my.score(X_test, y_test)

    # sklearn tree
    sk = SKTree(max_depth=max_depth)
    sk.fit(X_train, y_train)
    acc_sk = accuracy_score(y_test, sk.predict(X_test))

    print("=== Comparison ===")
    print("My Decision Tree:", acc_my)
    print("sklearn:", acc_sk)


# -----------------------------
# Simple CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    if args.demo:
        compare(max_depth=args.max_depth)
    else:
        print("Usage: python dt.py --demo")
