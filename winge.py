"""
Winge functions (w-shaped double-hinge functions)
useful for coercing toward Boolean. Expressable as
difference of convex functions
"""
import numpy as np


def hinge(x, slope):
    y = x * slope
    return y * (y > 0)


def winge_convex_left(x, t, slope):
    return hinge(x, -slope) + hinge(x, 1.0/t) - 0.25


def winge_convex(x, t, slope):
    return winge_convex_left(x, t, slope) + winge_convex_left(1-x, t, slope)


def winge_concave_left(x, t):
    return -hinge(x-t, 1.0/t) - 0.25


def winge_concave(x, t, alpha=0.0):
    w = winge_concave_left(x, t) + winge_concave_left(1-x, t)
    w += (- alpha * (abs(x-0.5) - abs(t-0.5)))
    return w


def winge(x, t, slope, alpha=0.0):
    assert 0 <= alpha <= 1
    convex = winge_convex(x, t, slope)
    concave = winge_concave(x, t, alpha=alpha*slope)
    return convex + concave


def winge_concave_linearized(x, x0, t, alpha=0.0):
    mask1 = x0 < t
    mask2 = t <= x0 < 0.5
    mask3 = 0.5 <= x0 <= (1-t)
    mask4 = x0 > (1-t)
    tang1 = -(1-x-t)/t - 0.5 - alpha * (0.5-x - abs(t-0.5))
    tang2 = - 0.5 - alpha * (0.5-x - abs(t-0.5))
    return tang1 * mask1 + tang2 * mask2


def hinge_2(x):
    h = 1.0 - x
    h[h < 0] = 0
    return h


def vmax(x, x_max):
    #subtracted 1
    result = x * 0.0
    w = x < 0
    result[w] = -(2.0*x[w])/x_max
    return result


def prox_hinge_2(c, rho):
    if c >= 1:
        return c
    if c < 1-(1.0/rho):
        return c+(1.0/rho)
    return 1.0


def prox_vmax(c, x_max, rho):
    alpha = 2.0/x_max
    b = 1.0
    rho_prime = rho / (alpha**2)
    return (prox_hinge_2(alpha * c + b, rho_prime) - b)/alpha


def ww(x):
    result = -x
    result[x > 0] = 0.0
    return result


def prox_ww(q, rho):
    a = -1.0/rho
    result = q * 0.0
    w1 = q < a
    result[w1] = q[w1] - a
    w2 = q > 0
    result[w2] = q[w2]
    return result


def test_prox_hinge_2(q, rho=0.8, doplot=True):
    from matplotlib import pylab as plt
    n = 100000
    p = np.linspace(-0.5, 0.5, n)
    res = ww(p) + 0.5 * rho * (p-q)**2
    index_min = np.argmin(res)
    p_min = p[index_min]
    plt.clf()
    plt.plot(p, res, color='red')
    print 'p_min: %s' % p_min
    plt.axvline(p_min, color='blue')
    qq = np.array([q])
    prox = prox_ww(qq, rho)
    plt.axvline(p_min, color='orange', linestyle='--')
    assert abs(prox-p_min) < 1e-5


class TestProximalHinge(TestCase):
    def test_prox_hinge(self):
        x = np.linspace(-4, 4, 100000)
        for rho in [0.5, 0.9, 1.0, 1.5, 1.7]:
            for c in np.linspace(-2, 2, 100):
                q = 0.5 * rho * (x-c)**2 + pu.hinge(x)
                arg_min = x[np.argmin(q)]
                min_prox = pu.prox_hinge(c, rho)
                diff = abs(arg_min-min_prox)
                tol = 1e-3
                assert diff < tol

    def test_prox_vmax(self):
        x = np.linspace(-4, 4, 100000)
        x_max = 1.3
        for rho in [0.5, 0.9, 1.0, 1.5, 1.7]:
            for c in np.linspace(-2, 2, 100):
                q = 0.5 * rho * (x-c)**2 + pu.vmax(x, x_max)
                arg_min = x[np.argmin(q)]
                min_prox = pu.prox_vmax(c, x_max, rho)
                diff = abs(arg_min-min_prox)
                tol = 1e-3
                assert diff < tol


