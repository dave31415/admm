import numpy as np

from derivatives import first_derivative_matrix


def first(n):
    return first_derivative_matrix(n).todense()


def dtd(n):
    d = first(n)
    return d.T * d


def ddt(n):
    d = first(n)
    return d * d.T


def one(n):
    return np.matrix(np.ones((n, n)))


def zero(n):
    return np.matrix(np.zeros((n, n)))


def eye(n):
    return np.matrix(np.eye(n))


def block(M, n_blocks):
    n = M.shape[0]
    m = n * n_blocks
    B = zero(m)
    for i in xrange(n_blocks):
        for j in xrange(n_blocks):
            B[i*n:(i+1)*n, j*n:(j+1)*n] = M

    return B


def block_diag(M, n_blocks):
    n = M.shape[0]
    m = n * n_blocks
    B = zero(m)
    for i in xrange(n_blocks):
        B[i*n:(i+1)*n, i*n:(i+1)*n] = M

    return B


def s_matrix(n_time_periods, n_skus):
    n = n_time_periods * n_skus
    return eye(n) - (n_skus-2.0) * block(eye(n_time_periods), n_skus)


def ds_matrix(n_time_periods, n_skus):
    DTD = dtd(n_time_periods)
    DTD_expanded = block_diag(DTD, n_skus)
    S = s_matrix(n_time_periods, n_skus)
    return DTD_expanded*S


def p_matrix(n_time_periods, n_skus, n_lines, rho=1.0):
    DS = ds_matrix(n_time_periods, n_skus)
    DS_expanded = block_diag(DS, n_lines)
    L = block(eye(n_time_periods*n_skus), n_lines)
    P = DS_expanded + rho*L
    return DS_expanded, L, P
