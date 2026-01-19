from typing import Tuple

import numpy as np

"""
Generate a synthetic regression dataset using a predefined nonlinear function.

Samples 20 scalar inputs x from a fixed uniform range using a given random seed,
computes corresponding targets y via a deterministic nonlinear expression, and
splits the data into training and test sets.

Parameters
----------
seed : int, optional
    Random seed for reproducibility.

Returns
-------
x_train : np.ndarray
    First 10 input samples, shape (10, 1).
x_test : np.ndarray
    Remaining 10 input samples, shape (10, 1).
y_train : np.ndarray
    Targets corresponding to x_train.
y_test : np.ndarray
    Targets corresponding to x_test.
"""

def dcgp_1(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(1, 3, size=(20,)).reshape(-1, 1)
    y = x ** 5 - np.pi * x ** 3 + x
    return x[:10], x[10:], y[:10], y[10:]


def dcgp_2(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(.1, 5, size=(20,)).reshape(-1, 1)
    y = x ** 5 - np.pi * x ** 3 + 2 * np.pi / x
    return x[:10], x[10:], y[:10], y[10:]


def dcgp_3(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(-.9, 1, size=(20,)).reshape(-1, 1)
    y = (np.e * x ** 5 + x ** 3) / (x + 1)
    return x[:10], x[10:], y[:10], y[10:]


def dcgp_4(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, size=(20,)).reshape(-1, 1)
    y = np.sin(np.pi * x) + 1 / x
    return x[:10], x[10:], y[:10], y[10:]


def dcgp_5(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(1, 3, size=(20,)).reshape(-1, 1)
    y = np.e * x ** 5 - np.pi * x ** 3 + x
    return x[:10], x[10:], y[:10], y[10:]


def dcgp_6(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(-2.1, 1, size=(20,)).reshape(-1, 1)
    y = (np.e * x ** 2 - 1) / (np.pi * (x + 2))
    return x[:10], x[10:], y[:10], y[10:]


def dcgp_7(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, size=(20,)).reshape(-1, 1)
    y = np.cos(np.pi * x) * np.sin(np.e * x)
    return x[:10], x[10:], y[:10], y[10:]
