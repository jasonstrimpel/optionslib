from scipy.stats import norm
from math import sqrt, pi
import numpy as np
np.set_printoptions(precision=8, suppress=True)


def N(x):
    """ Normal cumulative function

    :param x:
    :return:
    """
    return norm.cdf(x)


def phi(z):
    """ Phi helper function

    :param z:
    :return:
    """
    return np.exp(-0.5 * z * z) / (sqrt(2.0 * pi))


def call_value(s, k, r, q, t, vol):
    """ Black-Scholes call option

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
    :return: BS call option value
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return s * np.exp(-q * t) * N(d1) - k * np.exp(-r * t) * \
                                        N(d1 - vol * np.sqrt(t))


def put_value(s, k, r, q, t, vol):
    """ Black-Scholes put option

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
    :return: BS put option value
    """

    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return k * np.exp(-r * t) * N(-(d1 - vol * np.sqrt(t)) ) - s * \
                                        np.exp(-q * t) * N(-d1)