"""SABR model: simplified implied-volatility formula."""


def sigma_bs(m, alpha_t, beta, rho, nu, F_t):
    """Simplified SABR implied-vol approximation.

    m: log-moneyness log(K/F_t). alpha_t: instantaneous vol level.
    beta: CEV exponent. rho: vol/spot correlation. nu: vol-of-vol.
    """
    lam = nu / (alpha_t * (F_t ** (beta - 1)))
    slope = 0.5 * (beta - 1 + rho * lam)
    curvature = ((1 - beta) ** 2 + (2 - 3 * rho ** 2) * lam ** 2) / 12.0
    return alpha_t * (F_t ** (beta - 1)) * (1 + slope * m + curvature * m ** 2)
