import math
import numpy as np


def wienerprocess(timesteps, paths, maturity):
    """
    Parameters
    --------------
    timesteps (int) : number of timesteps until maturity
    paths (int) : number of paths to simulate
    maturity (double) : periods to which the simulation proceeds
 
    Returns
    ---------
    np.ndarray : timesteps+1 X paths Numpy array
 
    Usage
    -------
    W = wienerprocess(timesteps=100, paths=50, maturity=1.0)
    """
 
    wienerpath = np.zeros((timesteps, paths), np.double)
    sqrt_time_step = math.sqrt(maturity / timesteps)
 
    normrand = sqrt_time_step * np.random.randn(timesteps, paths)
 
    for i in xrange(1, timesteps):
        wienerpath[i, :] = wienerpath[i-1, :] + normrand[i-1, :]

    return wienerpath


def evolve(s0, drift, volatility, timesteps, paths, maturity):
    dt = maturity/timesteps
    W = wienerprocess(timesteps, paths, maturity)
    return s0 * np.exp((drift-(0.5*volatility*volatility)) * dt + (volatility * W))


def maturity(s0, drift, volatility, timesteps, paths, maturity, trials):
    dt = maturity/timesteps
    s = np.zeros((trials, paths))

    for i in xrange(1, trials):
        w = wienerprocess(timesteps, paths, maturity)
        s[i, :] = s0 * np.exp((drift-(0.5*volatility*volatility)) * dt +
                              (volatility * w[-1, :]))

    return s