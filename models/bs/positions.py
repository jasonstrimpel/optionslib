import greeks
import payoffs
import numpy as np


def position_payoff(s, k, r, q, t, vol, option_type, option_premium, option_trade_qty,
                  underlying_price, underlying_trade_qty):
    """

    """
    def fcn(k_, r_, t_, vol_, option_type_, option_premium_, option_trade_qty_):
        meth_name = '{0}_payoff'.format(option_type_)
        return getattr(payoffs, meth_name)(s, k_, r_, q, t_, vol_,
                                           option_premium_, option_trade_qty_)

    options = np.array([fcn(k, r, t, vol, option_type, option_premium, option_trade_qty) for
                        (k, r, t, vol, option_type, option_premium, option_trade_qty) in
                        zip(k, r, t, vol, option_type, option_premium, option_trade_qty)])

    underlying = payoffs.underlying_payoff(s, underlying_price, underlying_trade_qty)

    return {'position': sum(options) + underlying, 'components': np.vstack((options, underlying))}


def position_delta(s, k, r, q, t, vol, option_type, option_trade_qty, underlying_trade_qty):
    """

    """
    def fcn(k_, r_, t_, vol_, option_type_, option_trade_qty_):
        meth_name = '{0}_delta'.format(option_type_)
        return getattr(greeks, meth_name)(s, k_, r_, q, t_, vol_) * \
               option_trade_qty_

    option_deltas = np.array([fcn(k, r, t, vol, option_type, option_trade_qty) for
                              (k, r, t, vol, option_type, option_trade_qty) in zip(k, r, t, vol, option_type, option_trade_qty)])

    m, n = np.shape(option_deltas)
    underlying_deltas = np.array([greeks.underlying_delta()]) * np.ones((1, n))

    return {'position': sum(option_deltas) + underlying_deltas * float(underlying_trade_qty), \
           'components': np.vstack((option_deltas, underlying_deltas * float(underlying_trade_qty)))}


def position_gamma(s, k, r, q, t, vol, option_trade_qty):
    """

    """
    def fcn(k_, r_, t_, vol_, option_trade_qty_):
        return greeks.gamma(s, k_, r_, q, t_, vol_) * option_trade_qty_

    option_gammas = np.array([fcn(k, r, t, vol, option_trade_qty) for
                              (k, r, t, vol, option_trade_qty) in
                              zip(k, r, t, vol, option_trade_qty)])

    return {'position': sum(option_gammas), 'components': option_gammas}


def position_vega(s, k, r, q, t, vol, option_trade_qty):
    """

    """
    def fcn(s_, k_, r_, t_, vol_, option_trade_qty_):
        return greeks.vega(s_, k_, r_, q, t_, vol_) * option_trade_qty_

    option_vegas = np.array([fcn(s, k, r, t, vol, option_trade_qty) for
                              (k, r, t, vol, option_trade_qty) in
                              zip(k, r, t, vol, option_trade_qty)])

    return {'position': sum(option_vegas), 'components': option_vegas}


def position_theta(s, k, r, q, t, vol, option_type, option_trade_qty):
    """

    """
    def fcn(s_, k_, r_, vol_, option_type_, option_trade_qty_):
        meth_name = '{0}_theta'.format(option_type_)
        return getattr(greeks, meth_name)(s_, k_, r_, q, t, vol_) * \
               option_trade_qty_

    option_thetas = np.array([fcn(s, k, r, vol, option_type, option_trade_qty) for
                        (k, r, vol, option_type, option_trade_qty) in
                        zip(k, r, vol, option_type, option_trade_qty)])

    return {'position': sum(option_thetas), 'components': option_thetas}


def position_rho(s, k, r, q, t, vol, option_type, option_trade_qty):
    """

    """
    def fcn(s_, k_, t_, vol_, option_type_, option_trade_qty_):
        meth_name = '{0}_rho'.format(option_type_)
        return getattr(greeks, meth_name)(s_, k_, r, q, t_, vol_) * \
               option_trade_qty_

    option_rhos = np.array([fcn(s, k, t, vol, option_type, option_trade_qty) for
                        (k, t, vol, option_type, option_trade_qty) in
                        zip(k, t, vol, option_type, option_trade_qty)])

    return {'position': sum(option_rhos), 'components': option_rhos}