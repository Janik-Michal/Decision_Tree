import time
import sys
import os
import numpy as np

# --- Import bibliotek pomocniczych ---
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnTree

# --- Konfiguracja ścieżki do Twojego kodu ---
# Dodajemy folder 'src' do ścieżki, żeby Python widział Twoją paczkę
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from decision_tree.decision_tree import DecisionTreeClassifier as MyTree
except ImportError:
    print("BŁĄD: Nie znaleziono Twojego kodu.")
    print("Upewnij się, że uruchamiasz ten skrypt z głównego folderu projektu (Decision_Tree-main).")
    sys.exit(1)

def run_benchmark():
    # 1. Generowanie większego zbioru danych
    # 2000 próbek, 10 cech - to wystarczy, by zmęczyć Pythona, ale nie zawiesić go na minuty.
    print("Generowanie danych (2000 próbek)...")
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_informative=8, 
        n_redundant=0, 
        random_state=42,
        n_classes=2
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dane gotowe. Trening: {len(X_train)}, Test: {len(X_test)}")
    print("-" * 60)

    # Parametry drzewa
    MAX_DEPTH = 5  # Ograniczamy głębokość, żeby Python nie liczył tego w nieskończoność

    # ==========================================
    # MODEL 1: Twoja implementacja (Pure Python)
    # ==========================================
    print("Trenowanie Twojego modelu...")
    my_model = MyTree(max_depth=MAX_DEPTH, criterion='gini')
    
    # Pomiar czasu treningu
    start = time.perf_counter()
    my_model.fit(X_train, y_train)
    my_train_time = time.perf_counter() - start

    # Pomiar czasu predykcji
    start = time.perf_counter()
    my_preds = my_model.predict(X_test)
    my_infer_time_total = time.perf_counter() - start
    my_infer_time_per_img = (my_infer_time_total / len(y_test)) * 1000  # na milisekundy

    my_acc = accuracy_score(y_test, my_preds)

    # ==========================================
    # MODEL 2: Scikit-Learn (Optimized C)
    # ==========================================
    print("Trenowanie Scikit-Learn...")
    sk_model = SklearnTree(max_depth=MAX_DEPTH, criterion='gini', random_state=42)
    
    # Pomiar czasu treningu
    start = time.perf_counter()
    sk_model.fit(X_train, y_train)
    sk_train_time = time.perf_counter() - start

    # Pomiar czasu predykcji
    start = time.perf_counter()
    sk_preds = sk_model.predict(X_test)
    sk_infer_time_total = time.perf_counter() - start
    sk_infer_time_per_img = (sk_infer_time_total / len(y_test)) * 1000

    sk_acc = accuracy_score(y_test, sk_preds)

    # ==========================================
    # RAPORT KOŃCOWY
    # ==========================================
    print("\n")
    print("=" * 65)
    print(f"{'TEST SUMMARY: My Decision Tree vs Scikit-Learn':^65}")
    print("=" * 65)
    
    # Tabela wyników
    header = f"{'Model':<25} | {'Accuracy':<10} | {'Train Time':<12}"
    print(header)
    print("-" * 65)
    
    print(f"{'My Implementation':<25} | {my_acc:.4f}     | {my_train_time:.4f} s")
    print(f"{'Scikit-Learn (Baseline)':<25} | {sk_acc:.4f}     | {sk_train_time:.4f} s")
    print("-" * 65)
    
    print(f"\nInference (Prediction) Speed:")
    print(f"My Implementation : {my_infer_time_per_img:.4f} ms / sample")
    print(f"Scikit-Learn      : {sk_infer_time_per_img:.4f} ms / sample")
    
    # Podsumowanie dla kolegi
    print("-" * 65)
    print("INTERPRETATION:")
    print(f"1. Accuracy gap: {abs(my_acc - sk_acc)*100:.2f}% (Różnica w dokładności)")
    
    speedup = my_train_time / sk_train_time
    print(f"2. Speed factor: Scikit-Learn jest {speedup:.1f}x szybszy w treningu.")
    print("   (Jest to oczekiwane, ponieważ SKLearn używa C/Cython, a my Pythona)")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()