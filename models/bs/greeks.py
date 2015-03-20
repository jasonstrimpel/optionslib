from math import sqrt, pi
from scipy.stats import norm
import numpy as np


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


def call_delta(s, k, r, q, t, vol):
    """ Black-Scholes call delta

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
    :return: BS call delta
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return np.exp(-q*t)*N(d1)

	
def put_delta(s, k, r, q, t, vol):
    """ Black-Scholes put delta

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
    :return: BS put delta
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return -np.exp(-q*t)*N(-d1)


def underlying_delta():
    """ Black-Scholes underlying delta

    :return: underlying delta
    """
    return 1.0

	
def call_gamma(s, k, r, q, t, vol):
    """ Black-Scholes gamma

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
    :return: BS gamma
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return np.exp(-q*t) * phi(d1) / (s * vol * np.sqrt(t))
	
	
def put_gamma(s, k, r, q, t, vol):
    """ Black-Scholes gamma

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
    :return: BS gamma
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return np.exp(-q*t) * phi(d1) / (s * vol * np.sqrt(t))

	
def call_vega(s, k, r, q, t, vol, underlying_shares=100.):
    """ Black-Scholes vega

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
	:underlying_shares: shares of underlying
    :return: BS vega
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return (s * np.exp(-q*t) * phi(d1) * np.sqrt(t)) / underlying_shares

	
def put_vega(s, k, r, q, t, vol, underlying_shares=100.):
    """ Black-Scholes vega

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
	:underlying_shares: shares of underlying
    :return: BS vega
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return (s * np.exp(-q*t) * phi(d1) * np.sqrt(t)) / underlying_shares

	
def call_theta(s, k, r, q, t, vol):
    """ Black-Scholes call theta

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
	:days: theta per day
    :return: BS call theta
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    res = -np.exp(-q*t) * s * phi(d1) * vol * 0.5 / np.sqrt(t) - r *\
                k * np.exp(-r*t) * N(d2) + q * s * np.exp(-q*t) * N(d1)
    return res / 365.0

	
def put_theta(s, k, r, q, t, vol):
    """ Black-Scholes put theta

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
	:days: theta per day
    :return: BS put theta
    """
    d1 = (np.log(s/k) + (r - q + 0.5 * vol**2.0) * t) / ( vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    res = -np.exp(-q*t) * s * phi(d1) * vol * 0.5 / np.sqrt(t) + r * \
             k * np.exp(-r*t) * N(-d2) - q * s * np.exp(-q*t) * N(-d1)
    return res / 365.0


def call_rho(s, k, r, q, t, vol, underlying_shares=100.0):
    """ Black-Scholes call rho

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
	:underlying_shares: shares of underlying
    :return: BS call rho
    """
    d2 = (np.log(s/k) + (r - q - 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return (k * t * np.exp(-r*t) * N(d2)) / underlying_shares


def put_rho(s, k, r, q, t, vol, underlying_shares=100.0):
    """ Black-Scholes call rho

    :param s: underlying
    :param k: strike price
    :param r: rate
    :param q: dividend
    :param t: time to expiration
    :param vol: volatility
	:underlying_shares: shares of underlying
    :return: BS call rho
    """
    d2 = (np.log(s/k) + (r - q - 0.5 * vol**2.0) * t) / (vol * np.sqrt(t))
    return (-k * t * np.exp(-r*t) * N(-d2)) / underlying_shares