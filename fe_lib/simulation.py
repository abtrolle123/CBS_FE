"""GBM path simulation and Monte Carlo variance-reduction estimators."""

import numpy as np
from numpy import sqrt, exp
import scipy.stats as st


def simulate_gbm_path(S, r, delta, sigma, dt, n_steps, eps=None):
    """Simulate one GBM path of length n_steps.

    If `eps` is None, draws n_steps standard-normal shocks. Pass in an
    existing (or sign-flipped) `eps` array to build antithetic pairs.
    """
    if eps is None:
        eps = np.random.normal(size=n_steps)
    path = np.zeros(n_steps + 1)
    path[0] = S
    for j in range(n_steps):
        path[j + 1] = path[j] * exp((r - delta - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * eps[j])
    return path[1:]


def plain_mc_estimate(Y, confidence=0.95):
    """Mean and confidence interval for plain Monte Carlo samples Y."""
    mean = np.mean(Y)
    ci = st.norm.interval(confidence=confidence, loc=mean, scale=st.sem(Y))
    return mean, ci


def antithetic_estimate(Y, Ytilde, confidence=0.95):
    """Mean and CI for antithetic-variates pairs (Y, Ytilde)."""
    Y_ave = (Y + Ytilde) / 2
    mean = np.mean(Y_ave)
    ci = st.norm.interval(confidence=confidence, loc=mean, scale=st.sem(Y_ave))
    return mean, ci


def control_variate_estimate(Y, X, EX, confidence=0.95):
    """Mean, CI, and correlation for a control-variate estimator.

    Y: payoff samples. X: control samples. EX: known expectation of X.
    """
    correlation = np.corrcoef(Y, X)[0][1]
    b_star = np.cov(Y, X)[0][1] / np.var(X)
    Yb = Y - b_star * (X - EX)
    mean = np.mean(Yb)
    ci = st.norm.interval(confidence=confidence, loc=mean, scale=st.sem(Yb))
    return mean, ci, correlation
