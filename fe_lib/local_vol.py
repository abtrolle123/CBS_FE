"""Risk-neutral densities (Breeden-Litzenberger) and Dupire local volatility."""

import numpy as np

from fe_lib.black_scholes import BSM


def bs_call_grid(S, K_grid, delta, r, iv_grid, T):
    """BSM call prices across a strike grid, one implied vol per strike."""
    return np.array([BSM(S, K, delta, r, sigma, T, "call") for K, sigma in zip(K_grid, iv_grid)])


def risk_neutral_density(call_prices, dK):
    """Breeden-Litzenberger: risk-neutral density of S_T from d^2C/dK^2.

    Returns an array 2 shorter than `call_prices` (interior points only).
    """
    return (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / dK ** 2


def dupire_local_vol(S, K_grid, r, delta, iv_grid, T, dT, dK):
    """Dupire local variance from call prices at T and T + dT.

    Returns an array 2 shorter than `K_grid` (interior points only).
    """
    CT = bs_call_grid(S, K_grid, delta, r, iv_grid, T)
    CTdT = bs_call_grid(S, K_grid, delta, r, iv_grid, T + dT)

    dCdT = (CTdT[1:-1] - CT[1:-1]) / dT
    dCdK = (CT[2:] - CT[1:-1]) / dK
    d2CdK2 = (CT[2:] - 2 * CT[1:-1] + CT[:-2]) / dK ** 2

    return (dCdT + r * K_grid[1:-1] * dCdK) / (0.5 * K_grid[1:-1] ** 2 * d2CdK2)
