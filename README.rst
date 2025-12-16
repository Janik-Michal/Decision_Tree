

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

=============  
Decision-Tree  
=============  

This project is a **simple and transparent implementation of a Decision Tree classifier** based on the CART algorithm, written in pure Python using NumPy.

The model builds a **binary decision tree** by selecting feature thresholds that minimize impurity using **Gini impurity** or **Shannon entropy**. Tree construction is fully deterministic and supports configurable parameters such as **maximum depth** and **minimum samples per split**.

The implementation is designed for **educational purposes**, demonstrating how decision trees work internally. It is validated using **unit tests** and can be compared against scikit-learn’s `DecisionTreeClassifier`, achieving comparable accuracy on standard datasets.


---
 Installation

To run the project, first set up a Python virtual environment and install dependencies:

# Create virtual environment

python -m venv .venv

# Activate virtual environment

# Windows    .\.venv\Scripts\activate

# Linux / macOS    source .venv/bin/activate


# Install dependencies

pip install numpy scikit-learn pytest


---

Note
====

This project has been set up using **PyScaffold 4.6**. For details and usage
information on PyScaffold see [https://pyscaffold.org/](https://pyscaffold.org/).





