"""Backtesting: portfolio weights in, daily return series out.

The package exists to enforce one rule:

    Weights decided using data up to and INCLUDING day t are applied to the
    return from day t to day t+1.

The feature library deliberately does not apply this shift (see the docstring
in features/library.py). This package is the point of use, which confines
lookahead bias to a single function.
"""
