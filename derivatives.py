"""
Sparse derivative matrices
"""
import numpy as np
from itertools import chain
from scipy.sparse import spdiags, coo_matrix, dia_matrix
import numba
from admm.memo import memo


def first_derivative_matrix(n):
    """
    A sparse matrix representing the first derivative operator
    :param n: a number
    :return: a sparse matrix that applies the derivative operator
             to a numpy array or list to yield a numpy array
    """
    e = np.mat(np.ones((1, n)))
    return spdiags(np.vstack((-1*e, e)), range(2), n-1, n)


def first_derivative(x):
    """
    Derivative as a functional operator
    :param x: array with (N, n_time_periods) dimensions
    :return: first derivative iterable
             backward difference so will have 1 fewer elements
             derivative not defined for first element
    """
    return x[:, 1:] - x[:, 0:-1]


def get_a_factors(x, i):
    # These are all positive if sorted
    a0 = float(x[i + 1] - x[i])
    a2 = float(x[i] - x[i - 1])

    # Must be sorted nd neither cab be zero
    assert (a0 > 0) and (a2 > 0)
    a1 = a0 + a2

    scf = a1 / 2.0

    return [2.0 * scf / (a1 * a2), -2.0 * scf / (a0 * a2), 2.0 * scf / (a0 * a1)]


def get_values(x):
    n = len(x)
    values = []
    for i in range(1, n - 1):
        vals = get_a_factors(x, i)
        values.extend(vals)

    return np.array(values)

# get_values_jit = numba.jit('float64[:](float64[:])')(get_values)


@memo
def get_ij(m):
    i = list(chain(*[[_] * 3 for _ in range(m)]))
    j = list(chain(*[[_, _ + 1, _ + 2] for _ in range(m)]))
    return i, j


def second_derivative_matrix_nes_vals(x):
    """
    Get the second derivative matrix for non-equally spaced points
    :param : x numpy array of x-values
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivative of x
    assumes points are sorted
    """
    n = len(x)
    m = n - 2

    values = get_values(x)

    # i = list(chain(*[[_] * 3 for _ in range(m)]))
    # j = list(chain(*[[_, _ + 1, _ + 2] for _ in range(m)]))
    i, j = get_ij(m)

    return values, i, j, m, n


def second_derivative_matrix_nes(x):
    """
    Get the second derivative matrix for non-equally spaced points
    :param : x numpy array of x-values
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivative of x
    assumes points are sorted
    """
    values, i, j, m, n = second_derivative_matrix_nes_vals(x)
    d2 = coo_matrix((values, (i, j)), shape=(m, n))

    return dia_matrix(d2)
