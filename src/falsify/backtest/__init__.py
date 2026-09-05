"""Module 2: the backtester spine.

Weights in, honest daily return series out.

The whole module exists to enforce one rule:

    Weights decided using data up to and INCLUDING day t are applied to the
    return from day t to day t+1.

Module 1's feature library deliberately does NOT do this shift (see the
docstring in features/library.py). This package is the point of use, which is
why lookahead bias lives in exactly one function in this codebase.
"""
