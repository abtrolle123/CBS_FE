"""Numerical integration schemes used by the Fourier-transform pricers."""

import numpy as np


def GL_integration(xmax, N):
    """Gauss-Legendre integration points/weights, integrating 0 to xmax with N points."""
    uv, wgtv = np.polynomial.legendre.leggauss(N)
    uv = (uv + 1) * xmax / 2  # integration points
    wgtv = wgtv * xmax / 2  # integration weights
    return uv, wgtv


def trap_integration(xmin, xmax, N):
    """Trapezoid-rule integration points/weights, integrating xmin to xmax with N points."""
    uv = np.linspace(xmin, xmax, N)
    wgtv = np.ones(N)
    wgtv[[0, N - 1]] = 1 / 2
    wgtv = wgtv * (xmax - xmin) / (N - 1)
    return uv, wgtv


def default_gl_settings(integration_points=100, upper_bound=200):
    """The Gauss-Legendre setup repeated at the top of most pricing cells."""
    uv, wgtv = GL_integration(upper_bound, integration_points)
    return {"uv": uv, "wgt": wgtv}
