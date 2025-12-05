from setuptools import setup, find_packages

setup(
    name="decision-tree",
    version="0.1.0",
    description="Decision Tree implementation",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),  # automatycznie znajdzie decision_tree
    python_requires=">=3.10",
    install_requires=[
        # jeśli masz jakieś zależności, dodaj je tutaj
        # np. "numpy>=1.26.4",
        # "scikit-learn>=1.7.2",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
)
