from bs import values
from bs import positions
from scipy.optimize import fsolve
import numpy as np


def call_impvol(s, k, r, q, t, option_market_price):
    """ Call implied volatility

    """
    def fcn(v):
        return option_market_price-values.call_value(s, k, r, q, t, v)

    return min(fsolve(fcn, x0=0.5, col_deriv=True, xtol=0.00001, maxfev=5000)[0], 5.0)


def put_impvol(s, k, r, q, t, option_market_price):
    """ Put implied volatility

    """
    def fcn(v):
        return option_market_price-values.put_value(s, k, r, q, t, v)

    return min(fsolve(fcn, x0=0.5, col_deriv=True, xtol=0.00001, maxfev=5000)[0], 5.0)


def breakeven(s, k, r, q, t, vol, option_type, option_premium,
                                   option_trade_qty, underlying_price,
                                   underlying_trade_qty):
    x0 = s

    if option_type.size > 1:
        position_last = option_premium.sum()
        x0 = np.array([-1,1])*position_last+s

    def fcn(x0):
        net_position, _ = positions.position_payoff(x0, k, r, q, t, vol,
                                           option_type,
                                           option_premium,
                                           option_trade_qty, underlying_price,
                                           underlying_trade_qty)

        return net_position

    x = fsolve(fcn, x0=x0, xtol=0.01, maxfev=500)
    dollar_change = x-s
    percent_change = dollar_change/s

    return x, dollar_change, percent_change


def delta_neutral(s, k, r, q, t, vol, option_type, option_trade_qty, x0):

    def fcn(x0):
        net_position, _ = positions.position_delta(s, k, r, q, t, vol,
                                    option_type,
                                    option_trade_qty, x0)
        return net_position

    return fsolve(fcn, x0=x0, xtol=0.00001, maxfev=5000)