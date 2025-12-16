
.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/


=============  
Decision-Tree  
=============  

This project is a **simple and transparent implementation of a Decision Tree classifier** based on the CART algorithm, written in pure Python using NumPy.

The model builds a **binary decision tree** by selecting feature thresholds that minimize impurity using **Gini impurity** or **Shannon entropy**. Tree construction is fully deterministic and supports configurable parameters such as **maximum depth** and **minimum samples per split**.

The implementation is designed for **educational purposes**, demonstrating how decision trees work internally. It is validated using **unit tests** and can be compared against scikit-learn’s `DecisionTreeClassifier`, achieving comparable accuracy on standard datasets.


##  Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows**

```bash
.\.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy scikit-learn pytest
```


---

##  Running Tests

To run the unit tests:

```bash
pytest
```

---

##  Note

This project was generated using **PyScaffold 4.6**.

For more information about PyScaffold, visit:
[https://pyscaffold.org/](https://pyscaffold.org/)

---

##  License

MIT License

---












