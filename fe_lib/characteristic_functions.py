"""Characteristic functions of log(S_T) for the models covered in Lectures 1-3.

Every function has the signature ``charFuncX(u, tau, parameters)`` where
``parameters`` is a dict of the model's inputs (see each function for the
keys it reads). This lets the pricers in `fe_lib.pricing` dispatch on model
name without needing a special case per model.
"""

from numpy import sqrt, log, exp


def charFuncBSM(u, tau, parameters):
    """Characteristic function of log(S_T) under Black-Scholes-Merton."""
    x = log(parameters["S"])
    r = parameters["r"]
    delta = parameters["delta"]
    sigma = parameters["sigma"]
    return exp(1j * u * (x + (r - delta - 0.5 * sigma ** 2) * tau) - 0.5 * u ** 2 * sigma ** 2 * tau)


def charFuncSV(u, tau, parameters):
    """Characteristic function of log(S_T) under Heston (1993) stochastic volatility."""
    x = log(parameters["S"])
    kappa = parameters["kappa"]
    theta = parameters["theta"]
    rho = parameters["rho"]
    sigma = parameters["sigma"]
    v = parameters["v"]
    r = parameters["r"]
    delta = parameters["delta"]

    d = sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma ** 2 * (u * 1j + u ** 2))
    d = -d  # the little Heston trap
    g = (kappa - rho * sigma * u * 1j + d) / (kappa - rho * sigma * u * 1j - d)
    C = (r - delta) * u * 1j * tau + kappa * theta / sigma ** 2 * (
        (kappa - rho * sigma * u * 1j + d) * tau - 2 * log((1 - g * exp(d * tau)) / (1 - g))
    )
    D = (kappa - rho * sigma * u * 1j + d) / sigma ** 2 * (1 - exp(d * tau)) / (1 - g * exp(d * tau))
    return exp(C + D * v + 1j * u * x)


def charFuncJ(u, tau, parameters):
    """Characteristic function of log(S_T) under Merton (1976) jump-diffusion."""
    x = log(parameters["S"])
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    muJ = parameters["muJ"]
    sigmaJ = parameters["sigmaJ"]
    r = parameters["r"]
    delta = parameters["delta"]

    mubar = exp(muJ + 0.5 * sigmaJ ** 2) - 1
    C = (
        1j * u * (r - delta) - 0.5 * u * (u + 1j) * sigma ** 2
        + lamb * ((exp(1j * u * muJ - 0.5 * u ** 2 * sigmaJ ** 2) - 1) - 1j * u * mubar)
    ) * tau
    return exp(C + 1j * u * x)


def charFuncSVJ(u, tau, parameters):
    """Characteristic function of log(S_T) under Bates (1996) stochastic vol + jumps."""
    x = log(parameters["S"])
    kappa = parameters["kappa"]
    theta = parameters["theta"]
    rho = parameters["rho"]
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    muJ = parameters["muJ"]
    sigmaJ = parameters["sigmaJ"]
    v = parameters["v"]
    r = parameters["r"]
    delta = parameters["delta"]

    d = sqrt((rho * sigma * u * 1j - kappa) ** 2 + sigma ** 2 * (u * 1j + u ** 2))
    d = -d  # the little Heston trap
    g = (kappa - rho * sigma * u * 1j + d) / (kappa - rho * sigma * u * 1j - d)
    C = (r - delta) * u * 1j * tau + kappa * theta / sigma ** 2 * (
        (kappa - rho * sigma * u * 1j + d) * tau - 2 * log((1 - g * exp(d * tau)) / (1 - g))
    )
    D = (kappa - rho * sigma * u * 1j + d) / sigma ** 2 * (1 - exp(d * tau)) / (1 - g * exp(d * tau))
    Cmerton = (
        lamb * (exp(1j * u * muJ - 0.5 * u ** 2 * sigmaJ ** 2) - 1 - 1j * u * (exp(muJ + 0.5 * sigmaJ ** 2) - 1))
    ) * tau
    return exp(C + D * v + 1j * u * x + Cmerton)


def charFuncKou(u, tau, parameters):
    """Characteristic function of log(S_T) under Kou (2002) double-exponential jumps."""
    x = log(parameters["S"])
    sigma = parameters["sigma"]
    lamb = parameters["lamb"]
    p = parameters["p"]
    eta = parameters["eta"]
    r = parameters["r"]
    delta = parameters["delta"]

    mubar = p / (eta - 1) - (1 - p) / (eta + 1)
    C = (
        1j * u * (r - delta) - 0.5 * u * (u + 1j) * sigma ** 2
        + 1j * u * lamb * ((p / (eta - 1j * u) - (1 - p) / (eta + 1j * u)) - mubar)
    ) * tau
    return exp(C + 1j * u * x)


CHAR_FUNCTIONS = {
    "BSM": charFuncBSM,
    "SV": charFuncSV,
    "J": charFuncJ,
    "SVJ": charFuncSVJ,
    "Kou": charFuncKou,
}
