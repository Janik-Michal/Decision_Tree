import time
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnTree

# WAŻNE: Dostosuj ten import do nazwy swojego pliku!
# Jeśli Twój plik nazywa się dt.py, użyj: from dt import DecisionTreeClassifier, entropy, gini
from decision_tree.decision_tree import DecisionTreeClassifier, entropy, gini

# -------------------------------------------------------------------------
# 1. UNIT TESTS - METRYKI I LOGIKA
# -------------------------------------------------------------------------

def test_entropy_calculations():
    """Sprawdza poprawność obliczania entropii Shannona."""
    # Przypadek 1: Jednolita klasa (czystość idealna) -> Entropia 0
    y_pure = np.array([0, 0, 0, 0])
    assert entropy(y_pure) == 0.0

    # Przypadek 2: Równy podział 50/50 -> Entropia 1.0 (w logarytmie o podstawie 2)
    y_split = np.array([0, 0, 1, 1])
    assert np.isclose(entropy(y_split), 1.0)

    # Przypadek 3: Pusty zbiór -> 0
    assert entropy([]) == 0.0

def test_gini_calculations():
    """Sprawdza poprawność obliczania Gini Impurity."""
    # Przypadek 1: Jednolita klasa -> Gini 0
    y_pure = np.array([1, 1, 1])
    assert gini(y_pure) == 0.0

    # Przypadek 2: Równy podział dwóch klas -> Gini 0.5 (1 - (0.5^2 + 0.5^2))
    y_split = np.array([0, 0, 1, 1])
    assert np.isclose(gini(y_split), 0.5)

def test_model_initialization():
    """Sprawdza czy parametry są poprawnie przypisywane."""
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=4, criterion="entropy")
    assert clf.max_depth == 5
    assert clf.min_samples_split == 4
    assert clf.criterion == "entropy"

def test_single_class_fitting():
    """Model powinien od razu zwrócić liść, jeśli dane treningowe mają tylko jedną klasę."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 1, 1])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    # Drzewo powinno być samym korzeniem (liściem) z wartością 1
    assert clf.tree_.value == 1
    assert clf.tree_.left is None
    assert clf.tree_.right is None

def test_max_depth_constraint():
    """Sprawdza czy drzewo przestrzega ograniczenia głębokości."""
    # Generujemy dane, które normalnie wymagałyby głębokiego drzewa
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    max_d = 1
    clf = DecisionTreeClassifier(max_depth=max_d)
    clf.fit(X, y)
    
    # Funkcja pomocnicza do sprawdzania głębokości
    def get_depth(node):
        if node.value is not None:
            return 0
        return 1 + max(get_depth(node.left), get_depth(node.right))
    
    actual_depth = get_depth(clf.tree_)
    assert actual_depth <= max_d

# -------------------------------------------------------------------------
# 2. INTEGRATION & COMPARISON TESTS (Styl Twojego kolegi)
# -------------------------------------------------------------------------

@pytest.fixture
def iris_data():
    """Fixture dostarczający dane Iris."""
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)

def test_comparison_with_sklearn_and_report(iris_data, capsys):
    """
    Porównuje Twoją implementację ze Scikit-Learn.
    Generuje raport tekstowy widoczny przy użyciu flagi `pytest -s`.
    """
    X_train, X_test, y_train, y_test = iris_data

    # --- 1. Twoja implementacja ---
    start_time = time.time()
    my_model = DecisionTreeClassifier(max_depth=3, criterion='gini')
    my_model.fit(X_train, y_train)
    my_train_time = time.time() - start_time

    start_time = time.time()
    my_preds = my_model.predict(X_test)
    my_inference_time = (time.time() - start_time) * 1000 / len(y_test) # ms na próbkę
    my_acc = accuracy_score(y_test, my_preds)

    # --- 2. Scikit-Learn (Baseline) ---
    start_time = time.time()
    sk_model = SklearnTree(max_depth=3, criterion='gini', random_state=42)
    sk_model.fit(X_train, y_train)
    sk_train_time = time.time() - start_time

    start_time = time.time()
    sk_preds = sk_model.predict(X_test)
    sk_inference_time = (time.time() - start_time) * 1000 / len(y_test)
    sk_acc = accuracy_score(y_test, sk_preds)

    # --- 3. Generowanie Raportu (Wypisywanie na konsolę) ---
    print("\n")
    print("="*60)
    print(f"{'TEST SUMMARY: Custom DT vs Scikit-Learn':^60}")
    print("="*60)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Training (s)':<12}")
    print("-" * 60)
    print(f"{'Custom Implementation':<25} | {my_acc:.4f}     | {my_train_time:.5f} s")
    print(f"{'Scikit-Learn (Baseline)':<25} | {sk_acc:.4f}     | {sk_train_time:.5f} s")
    print("-" * 60)
    print(f"Custom Inference: {my_inference_time:.4f} ms/sample")
    print(f"Sklearn Inference: {sk_inference_time:.4f} ms/sample")
    print("="*60)

    # --- 4. Asercje (Warunki zaliczenia testu) ---
    
    # Dokładność Twojego modelu nie powinna być drastycznie gorsza od sklearn
    # (dopuszczamy margines błędu np. 10% dla prostej implementacji)
    assert my_acc >= (sk_acc - 0.10), "Twoje drzewo ma znacznie gorszą dokładność niż Sklearn!"
    
    # Dokładność na zbiorze Iris powinna być wysoka (>85%)
    assert my_acc > 0.85, "Dokładność modelu jest zbyt niska dla prostego zbioru Iris."