from scipy.sparse.linalg import splu
from scipy.sparse import spdiags, identity, csc_matrix
import numpy as np
from scipy.sparse.csr import csr_matrix
import scipy
from admm.memo import memo_array, memo


# TODO, check out these alternatives for multidimensional matrix mult
# http://tinyurl.com/jglhjh6


def time_weighting_old(t):
    return 5.0 / (5.0 + np.log10(10.0 + t) - 1.0)


def time_weighting(n_time_periods):
    return 1 - 0.2 * np.linspace(0, 1, n_time_periods)


def time_weight_array(shape):
    return np.outer(time_weighting(shape[0]), np.ones(shape[1]))


def make_positive(array):
    array[array < 0] = 0


def make_less_than_or_equal_to_one(array):
    array[array > 1] = 1.0


def make_between_zero_and_one(array):
    make_positive(array)
    make_less_than_or_equal_to_one(array)


def matrix_multiply_multidimensional_matrices(matrix, array, dim_array):
    n_dims = len(array.shape)
    assert dim_array < n_dims
    assert isinstance(matrix, csr_matrix) or isinstance(matrix, scipy.sparse.dia.dia_matrix)
    if n_dims == 1:
        return matrix * array

    if n_dims == 2:
        if dim_array == 0:
            return matrix * array
        else:
            return array * matrix.T

    n = array.shape[dim_array]
    other_dims = list(set(array.shape) - {n})
    m = other_dims[0] * other_dims[1]
    array_swapped = array.swapaxes(0, dim_array)
    array_swapped_shape = array_swapped.shape
    array_swapped_reshaped = array_swapped.reshape(n, m)
    result = matrix * array_swapped_reshaped
    return result.reshape(array_swapped_shape).swapaxes(0, dim_array)


def expand_array_over_new_dimension(arr, axis, length):
    new_shape = list(arr.shape)
    new_shape.insert(axis, length)
    new_strides = list(arr.strides)
    new_strides.insert(axis, 0)
    return np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)


def expand_array_first_row(arr, length):
    n = len(arr)
    new_array = np.zeros((length, n))
    new_array[0, :] = arr
    return new_array


def solve_uu(b, beta, dimension):
    """
    Solve the linear equation M x = b
    where M is a matrix defined by the Kronecker product
    (I + beta U U^T) x I x I (if dimension is 0)
    I x (I + beta U U^T) x I (if dimension is 1)
    I x I x (I + beta U U^T) (if dimension is 2)
    U is the vector with all 1's
    :param b: the right hand side
    :param beta: any number except -1/n
        where n is the size of that dimension
    :param dimension: dimension in the array
    :return: the solution for x
    """
    n = b.shape[dimension]
    tol = 1e-9
    if abs(beta - (-1.0 / n)) < tol:
        raise ValueError('beta too close to singular value')

    f = beta / (1.0 + beta * n)
    b_tmp = b.swapaxes(0, dimension)
    return (b_tmp - f * b_tmp.sum(0)).swapaxes(0, dimension)


def first_derivative_matrix(n):
    """
    A sparse matrix representing the first derivative operator
    :param n: a number
    :return: a sparse matrix that applies the derivative operator
             to a numpy array or list to yield a numpy array
    """
    e = np.mat(np.ones((1, n)))
    return spdiags(np.vstack((-1 * e, e)), range(2), n - 1, n)


def get_dtd(n):
    """
    Return a sparse representation of the D^T D matrix
    where D is the first derivative operator
    :param n: positive integer, matrix will be n x n
    :return: D^T D matrix
    """
    d = first_derivative_matrix(n)
    return d.T * d


class DtDSolver(object):
    """ Returns a callable object which
        solves the linear system M x = b
        where M is a matrix defined by the Kronecker product
        (I + beta D^T D) x I x I (if dimension is 0)
        I x (I + beta D^T D) x I (if dimension is 1)
        I x I x (I + beta D^T D) (if dimension is 2)
        and solver has already been created by get_dtd_solver()
        Correctly handles the case where the array b
        is multidimensional. Get numpy to broadcast
        the solution properly over the other dimensions
        :param b: the right hand side of equation
        :param solver: the solver calculated with get_dtd_solver()
        :param dimension: dimension corresponding to the
            matrix M
        :return:
        """

    def __init__(self, n, beta, initially_constrained=False):
        """
        :param n: size of matrix
        :param beta: any number
        :return:
        """
        self.n = n
        self.beta = beta
        self.solver = self._get_dtd_solver(initially_constrained=initially_constrained)

    def __call__(self, b, dimension):
        b_shape = b.shape
        n_dims = len(b_shape)

        if n_dims == 1:
            return self.solver.solve(b)

        if n_dims == 2:
            if dimension == 0:
                return self.solver.solve(b)
            return self.solver.solve(b.T).T

        # put the dimension in the first index and flatten
        # over the others. This allows numpy to broadcast
        # the solve method. Then reverse this operation
        # in order to match original shape

        a = b.swapaxes(0, dimension)
        a_shape = a.shape
        n_dims_flat = np.prod(a_shape[1:])
        n_first = a_shape[0]
        a_flatter = a.reshape([n_first, n_dims_flat])
        solution = self.solver.solve(a_flatter)
        return solution.reshape(a_shape).swapaxes(0, dimension)

    def _get_dtd_solver(self, initially_constrained=False):
        """
        Return a solver with a solve() method which
        solves the linear systems
        M x = b with M = I + beta D^T D
        where D is the first derivative matrix
        :param n: positive integer, size of D^T D matrix is n x n
        :param beta: any number
        :return:
        """
        dtd = get_dtd(self.n)
        mat = identity(self.n) + self.beta * dtd
        mat = mat.tocsc()
        if initially_constrained:
            # add 1 to first element which models an initial constraint
            add_matrix = self.beta * csc_matrix(([1], ([0], [0])), shape=(self.n, self.n))
            mat = mat + add_matrix

        solver = splu(mat)
        return solver


def eye_plus_ata_solver(matrix, beta):
    """
    Function which takes a matrix A and returns a solver
    (i.e. x=solve(y)) corresponding to the system
    (I + A^T A) x = y
    :param matrix: a (possibly) dense matrix or numpy array
    :return: solver which takes a vector of size A.shape[1]
             and returns the solution vector also of size A.shape[1]
    """
    matrix_sparse = csc_matrix(matrix)
    n = matrix_sparse.shape[1]
    mat = identity(n, format='csc') + beta * (matrix_sparse.T * matrix_sparse)
    solver = splu(mat)
    return solver


def project_to_cardinality_one(arr):
    # replace a 3D array with all zeros except for the
    # value which is the maximum of the 3rd dimension
    # can make this faster using
    # numpy expressions
    # also limit to range [0,1]
    shape = arr.shape
    result = arr * 0.0
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            k = np.argmax(arr[i, j, :])
            result[i, j, k] = arr[i, j, k]
    result[result < 0] = 0.0
    result[result > 1] = 1.0
    return result


def project_to_cardinality_n(arr, n):
    # replace a 3D array with all zeros except for the
    # value which is the maximum of the 3rd dimension
    # can make this faster using
    # numpy expressions
    # also limit to range [0,1]
    shape = arr.shape
    result = arr * 0.0
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            so = np.argsort(-arr[i, j, :])
            for l in xrange(n):
                k = so[l]
                result[i, j, k] = arr[i, j, k]

    result[result < 0] = 0.0
    result[result > 1] = 1.0
    return result


def soft_thresholding(array, kappa):
    # Soft thresholding operator, S_k(array)
    # is the prox function of lam * abs(x)
    # with k = lam/rho

    # turn it into an array if it's a scalar
    k = array*0.0 + kappa
    assert k.min() >= 0

    greater = array > k
    less = array < -k
    result = array * 0.0
    result[greater] = array[greater] - k[greater]
    result[less] = array[less] + k[less]
    return result


def prox_zero_hinge(q, k):
    # is the prox function of lam * zh(x)
    # where zh = -x for x< 0 and 0 x >= 0
    # with k = lam/rho
    result = q * 0.0
    w1 = q < (-k)
    result[w1] = q[w1] + k
    w2 = q > 0
    result[w2] = q[w2]
    return result


def multiply_3D_dim_zero_slow(matrix, array):
    shape = array.shape
    final_shape = (matrix.shape[0], array.shape[1], array.shape[2])
    result = np.zeros(final_shape)
    for i in xrange(shape[1]):
        for j in xrange(shape[2]):
            result[:, i, j] = matrix * array[:, i, j]
    return result.reshape(final_shape)


def multiply_3D_dim_zero(matrix, array):
    shape = array.shape
    final_shape = (matrix.shape[0], array.shape[1], array.shape[2])
    array_reshaped = array.reshape(shape[0], shape[1] * shape[2])
    return (matrix * array_reshaped).reshape(final_shape)


def get_tie_breaker_array(n_skus, seed):
    np.random.seed(seed)
    return np.random.random(n_skus)


def max_tie_breaker(p, tie_breakers):
    # get an array that is 1 if the value is the largest
    # in it's (time, line,:) column
    # as long as the max is > 0
    # break ties for a tls variable along the sku direction
    # based on an input tie_breaker array
    # doesn't need to be fast as it only gets called
    # once per dccp iter
    shape = p.shape
    results = np.zeros(shape, dtype=int)
    for t in xrange(shape[0]):
        for l in xrange(shape[1]):
            arr = p[t, l, :]
            index_max = np.argmax(arr)
            if arr[index_max] <= 0:
                continue
            w_max = np.where(arr == arr[index_max])[0]
            if len(w_max) > 0:
                breakers = tie_breakers[w_max]
                arg_max = np.argmax(breakers)
                index_max = w_max[arg_max]
            results[t, l, index_max] = 1
    return results


def cummulative_solve_1d(y):
    # solve for x; (I + C) x = y
    # x_{i} = (y_i - y_{i-1} + x_{i-1} )/2
    x = 0 * y
    x[0] = y[0] / 2.0
    n = len(y)
    for i in xrange(1, n):
        x[i] = (y[i] - y[i - 1] + x[i - 1]) / 2.0
    return x


def cumulative_solve(y):
    # solve (I + C) x = y for each sku independently
    n_time_periods, n_skus = y.shape
    x = y * 0.0
    for s in xrange(n_skus):
        x[:, s] = cummulative_solve_1d(y[:, s])
    return x


def cumulative_matrix(n):
    return np.matrix(np.tril(np.ones((n, n))))


@memo
def ctc_solver_memoized(args):
    return CtcSolver(args)


class CtcSolver:
    # Solve the system given by (scaling*I + C.T * C)* x = y
    # where C is the cumulative matrix and scaling is any number
    # i.e. the matrix that is equivalent to numpy.cumsum(x) operator

    def __init__(self, args):
        n_dimensions, scaling = args
        cm = cumulative_matrix(n_dimensions)
        matrix = scaling * np.eye(n_dimensions) + cm.T * cm
        self.cholesky_factorization = scipy.linalg.cho_factor(matrix)

    def __call__(self, y):
        return scipy.linalg.cho_solve(self.cholesky_factorization, y)


def backward_cumulative(x):
    return x[::-1, :].cumsum(0)[::-1]


def cholesky_tridiag(dd, ee):
    # in place cholesky factorization of a tridiagonal matrix
    # which has dd (n dimensional) on the main diagonal
    # and ee (n-1 dimensional) on the neighboring diagonals
    # ftp://ftp.cs.utexas.edu/pub/techreports/tr02-23.pdf
    # Parallel Cholesky Factorization of a Block Tridiagonal Matrix
    # Thuan D. Cao and Robert A. van de Geijn

    d = dd.copy()
    e = ee.copy()
    # inplace calculation
    n = len(d)
    for i in xrange(n-1):
        d[i] = np.sqrt(d[i])
        e[i] /= d[i]
        d[i+1] -= e[i]**2
    d[-1] = np.sqrt(d[-1])
    return d, e


def mult_by_g_matrix(x):
    # fast operators that represent the matrix
    # multiplication by G where G is the inverse of
    # the cumulative matrix
    prev = np.zeros(x.shape)
    prev[1:] = x[:-1]
    return x - prev


def mult_by_g_t_matrix(x):
    # fast operators that represent the matrix
    # multiplication by G.T where G.T is the transpose of
    # the inverse of the cumulative matrix
    next = np.zeros(x.shape)
    next[:-1] = x[1:]
    return x - next


def gtg_solver(y, beta):
    # solver for M x = y
    # M = beta * I + G.T G
    # where G is the inverse of cumulative matrix
    # Uses Thomas algorithm

    n = len(y)
    diag = 2.0 * np.ones(n)
    diag[-1] = 1.0
    diag += beta
    off_diag = -1 * np.ones(n-1)
    return tri_diagonal_matrix_solver(off_diag, diag, off_diag, y)


def dtd_solver(y, beta):
    # solver for M x = y
    # M = beta * I + D.T D
    # where D is the first derivative matrix
    # Uses Thomas algorithm

    n = len(y)
    diag = 2.0 * np.ones(n)
    diag[-1] = 1.0
    diag[0] = 1.0
    diag += beta
    off_diag = -1 * np.ones(n-1)
    return tri_diagonal_matrix_solver(off_diag, diag, off_diag, y)


def tri_diagonal_matrix_solver(lower_diagonal, diagonal,
                               upper_diagonal, right_hand_side):
    """
    Thomas Algorithm
    tri-diagonal matrix solver
    http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = len(diagonal)

    diag = diagonal.copy()
    lower_diag = np.zeros(n)
    upper_diag = np.zeros(n)
    lower_diag[1:] = lower_diagonal
    upper_diag[:-1] = upper_diagonal
    rhs = right_hand_side.copy()

    for i in xrange(1, n):
        rat = lower_diag[i]/diag[i-1]
        diag[i] = diag[i] - rat * upper_diag[i-1]
        rhs[i] = rhs[i] - rat * rhs[i-1]

    x = rhs * 0
    x[-1] = rhs[-1]/diag[-1]

    for i in xrange(n-2, -1, -1):
        x[i] = (rhs[i]-upper_diag[i] * x[i+1])/diag[i]

    return x


def switch_on(x, sigma, beta):
    xx = abs(x)/float(sigma)
    xb = xx ** beta
    return xb/(1.0 + xb)


def switch_off(x, sigma, beta):
    return 1.0 - switch_on(x, sigma, beta)



