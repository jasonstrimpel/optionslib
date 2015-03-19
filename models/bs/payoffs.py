"""
Module contains methods for computing option and underlying payoffs


"""

import values


def call_payoff(s, k, r, q, t, vol, option_premium, option_trade_qty,
                underlying_shares=100.0):
    """ Returns BS payoff in dollars

    """
    return (values.call_value(s, k, r, q, t, vol) - option_premium) * \
            option_trade_qty * underlying_shares


def put_payoff(s, k, r, q, t, vol, option_premium, option_trade_qty,
               underlying_shares=100.0):
    """ Returns BS payoff in dollars

    """
    return (values.put_value(s, k, r, q, t, vol) - option_premium) * \
            option_trade_qty * underlying_shares


def underlying_payoff(s, underlying_price, underlying_trade_qty):
    """ Returns payoff of underlying

    """
    return (s - underlying_price) * float(underlying_trade_qty)
