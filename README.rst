.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/decision-tree.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/decision-tree
    .. image:: https://readthedocs.org/projects/decision-tree/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://decision-tree.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/decision-tree/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/decision-tree
    .. image:: https://img.shields.io/pypi/v/decision-tree.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/decision-tree/
    .. image:: https://img.shields.io/conda/vn/conda-forge/decision-tree.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/decision-tree
    .. image:: https://pepy.tech/badge/decision-tree/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/decision-tree
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/decision-tree

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=============
Decision-Tree
=============


    This project is a simple and transparent implementation of a Decision Tree classifier based on the CART algorithm, written in pure Python using NumPy.

The model builds a binary decision tree by selecting feature thresholds that minimize impurity using Gini impurity or Shannon entropy. Tree construction is fully deterministic and supports configurable parameters such as maximum depth and minimum samples per split.

The implementation is designed for educational purposes and demonstrates how decision trees work internally. It is validated using unit tests and compared against scikit-learn’s DecisionTreeClassifier, achieving comparable accuracy on standard datasets.




.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.


