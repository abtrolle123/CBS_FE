"""Fourier-inversion option pricing, implied vols, and calibration.

`price_and_iv` and `calibration_residuals` both build on the same inner
pricing loop (`_price_grid`).
"""

import math

import numpy as np
from numpy import sqrt, log, exp, pi, real
import matplotlib.pyplot as plt

from fe_lib.black_scholes import BSM_IV
from fe_lib.characteristic_functions import CHAR_FUNCTIONS


def _price_grid(parameters, settings):
    """Fourier-inversion call prices for every (strike, maturity) pair.

    settings keys: uv, wgt (integration points/weights), strike (array,
    shape (n_strikes, n_maturities) or broadcastable), tauv (maturities),
    model (key into CHAR_FUNCTIONS), plot_integrand ('Y'/'N', optional).
    """
    S = parameters["S"]
    r = parameters["r"]
    delta = parameters["delta"]

    uv = settings["uv"]
    wgt = settings["wgt"]
    strike = np.atleast_2d(settings["strike"])
    N, M = strike.shape
    tauv = np.atleast_1d(settings["tauv"])
    char_func = CHAR_FUNCTIONS[settings["model"]]
    plot_integrand = settings.get("plot_integrand", "N")

    price_fit = np.zeros((N, M))
    for m in range(M):
        tau = tauv[m]
        charFct1 = char_func(uv - 1j, tau, parameters) / (S * exp((r - delta) * tau))
        charFct2 = char_func(uv, tau, parameters)
        for n in range(N):
            K = strike[n, m]
            integrand1 = real(exp(-1j * uv * log(K)) * charFct1 / (1j * uv))
            P1 = 0.5 + 1 / pi * sum(wgt * integrand1)
            integrand2 = real(exp(-1j * uv * log(K)) * charFct2 / (1j * uv))
            P2 = 0.5 + 1 / pi * sum(wgt * integrand2)
            price_fit[n, m] = S * exp(-delta * tau) * P1 - K * exp(-r * tau) * P2

            if plot_integrand == "Y":
                plt.plot(uv, integrand1, uv, integrand2)
                plt.gca().legend(("integrand 1", "integrand 2"))
                plt.show()

    return price_fit


def price_and_iv(parameters, settings, iv_guess=0.20):
    """Fourier-inversion call prices and their BSM implied vols.

    Returns (price_fit, IV_fit), both shaped (n_strikes, n_maturities).
    """
    S = parameters["S"]
    r = parameters["r"]
    delta = parameters["delta"]
    strike = np.atleast_2d(settings["strike"])
    tauv = np.atleast_1d(settings["tauv"])
    N, M = strike.shape

    price_fit = _price_grid(parameters, settings)
    IV_fit = np.zeros((N, M))
    for m in range(M):
        tau = tauv[m]
        for n in range(N):
            K = strike[n, m]
            IV_fit[n, m] = BSM_IV(S, K, delta, r, iv_guess, tau, "call", price_fit[n, m], 1e-5, 1000, "N")

    return price_fit, IV_fit


def calibration_residuals(param_vector, param_keys, fixed_parameters, settings):
    """Vega-scaled pricing errors, for use as a `scipy.optimize.least_squares` objective.

    Works for any model (via settings["model"]) and any subset of free
    parameters (param_keys).
    """
    parameters = dict(fixed_parameters)
    parameters.update(zip(param_keys, param_vector))
    price_fit = _price_grid(parameters, settings)
    err = (price_fit - settings["price"]) / settings["vega"]
    return err.ravel()


def merton_series_price(parameters, settings):
    """Merton (1976) call price as a Poisson-weighted sum of BSM prices (closed form).

    settings keys: strike, tauv (scalars here), truncation (# terms in the sum).
    """
    from fe_lib.black_scholes import BSM

    S = parameters["S"]
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    muJ = parameters["muJ"]
    sigmaJ = parameters["sigmaJ"]
    r = parameters["r"]
    delta = parameters["delta"]

    K = settings["strike"]
    tau = settings["tauv"]
    truncation = settings["truncation"]

    mu_bar = exp(muJ + 0.5 * sigmaJ ** 2) - 1
    lamb_prime = lamb * (1 + mu_bar)
    price = 0
    for k in range(truncation):
        sigma_k = sqrt(sigma ** 2 + k / tau * sigmaJ ** 2)
        r_k = r - mu_bar * lamb + k / tau * (muJ + 0.5 * sigmaJ ** 2)
        prob = exp(-lamb_prime * tau) * (lamb_prime * tau) ** k / math.factorial(k)
        price = price + prob * BSM(S, K, delta, r_k, sigma_k, tau, "call")
    return price
