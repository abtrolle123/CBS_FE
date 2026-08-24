"""Black-Scholes-Merton pricing, greeks, and implied volatility."""

import numpy as np
from numpy import sqrt, log, exp
from scipy.stats import norm


def _d1_d2(S, K, delta, r, sigma, T):
    d1 = (log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def BSM(S, K, delta, r, sigma, T, optionType):
    """BSM option price.

    S: underlying, K: strike, delta: dividend yield, r: risk-free rate,
    sigma: diffusion parameter, T: expiration, optionType: 'call' or 'put'.
    """
    d1, d2 = _d1_d2(S, K, delta, r, sigma, T)
    if optionType == "call":
        return S * exp(-delta * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-delta * T) * norm.cdf(-d1)


def BSM_greeks(S, K, delta, r, sigma, T, optionType):
    """BSM Delta, gamma, vega, vanna, volga. See http://www.nematrian.com/BlackScholesGreeksVanillaCalls."""
    d1, d2 = _d1_d2(S, K, delta, r, sigma, T)
    gamma = exp(-delta * T) * norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * exp(-delta * T) * norm.pdf(d1) * sqrt(T)
    vanna = -d2 * exp(-delta * T) * norm.pdf(d1) / sigma
    volga = d1 * d2 * S * exp(-delta * T) * norm.pdf(d1) * sqrt(T) / sigma
    if optionType == "call":
        Delta = exp(-delta * T) * norm.cdf(d1)
    else:
        Delta = -exp(-delta * T) * norm.cdf(-d1)
    return Delta, gamma, vega, vanna, volga


def BSM_vega(S, K, delta, r, sigma, T):
    d1, _ = _d1_d2(S, K, delta, r, sigma, T)
    return S * exp(-delta * T) * norm.pdf(d1) * sqrt(T)


def BSM_IV(S, K, delta, r, sigma, T, optionType, truePrice, tol, max_iterations, printOutput):
    """Implied volatility via Newton-Raphson, starting from `sigma`."""
    diff = None
    for i in range(max_iterations):
        diff = BSM(S, K, delta, r, sigma, T, optionType) - truePrice
        if abs(diff) < tol:
            break
        sigma = sigma - diff / BSM_vega(S, K, delta, r, sigma, T)
    if printOutput == "Y":
        print(f"found on {i}th iteration")
        np.set_printoptions(precision=4, suppress=False)
        print("difference is equal to ", np.array2string(diff))
    return float(sigma)
