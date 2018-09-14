import numpy as np
from scipy.sparse import spdiags, identity, lil_matrix, csc_matrix, coo_matrix, dia_matrix
from time import time
from admm.derivatives import second_derivative_matrix_nes, second_derivative_matrix_nes_vals
from admm.proximal_utils import eye_plus_ata_solver, soft_thresholding


def get_matrices_slow(x, alpha):
    # Make this simpler/faster
    n = len(x)
    d2 = second_derivative_matrix_nes(x)

    mat = lil_matrix((2 * n - 2, n))
    mat[0:n, :] = identity(n)
    mat[n:, :] = alpha * d2
    mat = csc_matrix(mat)
    return d2, mat


def get_matrices(x, alpha):
    # Make this simpler/faster
    values, i, j, m, n = second_derivative_matrix_nes_vals(x)
    d2 = coo_matrix((values, (i, j)), shape=(m, n))
    d2 = dia_matrix(d2)

    values = np.array(values)

    values = np.concatenate((np.ones(n), alpha * values))
    i = np.concatenate((np.arange(n), n + np.array(i)))
    j = np.concatenate((np.arange(n), np.array(j)))

    mat = coo_matrix((values, (i, j)), shape=(m + n, n))
    mat = dia_matrix(mat)

    return d2, mat


def get_vars(x, data, alpha, randomize=True):
    n = len(data)
    b = np.zeros(2 * n - 2)
    b[0:n] = data

    if randomize:
        np.random.seed(42)
        v = np.random.randn(2 * n - 2)
    else:
        v = b.copy() * 0.0

    u = v * 0.0

    # Make this simpler/faster
    d2, mat = get_matrices(x, alpha)

    return v, u, b, mat, d2


def iterate(niter, mat, b, v, u, solver, rho, break_early_iter, break_early_resid_tol):
    mat_t = mat.T
    y_last = None
    change = 1e6
    for i in range(niter):
        rhs = mat_t * (b + v - u)
        y = solver.solve(rhs)
        mat_y = mat * y
        xx = mat_y - b + u
        v = soft_thresholding(xx, 1./rho)
        resid = mat_y - v - b
        if y_last is not None:
            change = abs(y - y_last).max()

        y_last = y

        # print(i, abs(resid).max())
        u += resid

        if i % 100 == 0:
            res_max = abs(resid).max()
            print('i=%s, log(res_max): %0.4f, log(change): %0.4f' % (i,
                                                               np.log10(res_max),
                                                               np.log10(change)))

        if i >= break_early_iter and change < 1e-9:
            res_max = abs(resid).max()
            if res_max <= break_early_resid_tol:
                print('break at iter: %s, res_max: %s, change: %s' % (i, res_max, change))
                break

    print("iter: %s, max_resid: %s, change: %s" % (i, abs(resid).max(), change))
    return v


def trend_filter(x, data, alpha, rho, niter,
                 break_early_iter=10,
                 break_early_resid_tol=1e-9,
                 randomize=True):

    start = time()
    v, u, b, mat, d2 = get_vars(x, data, alpha, randomize=randomize)

    solver = eye_plus_ata_solver(d2, alpha**2)
    v = iterate(niter, mat, b, v, u, solver, rho, break_early_iter, break_early_resid_tol)

    n = len(x)
    y_final = v[0:n] + data

    run_time = time() - start
    print('Time: %s sec, %s per iter' % (run_time, run_time/niter))
    return y_final
