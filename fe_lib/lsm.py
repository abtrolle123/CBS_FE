"""Heston path simulation and the Longstaff-Schwartz (LSM) American option pricer."""

import numpy as np
from scipy.stats import norm, skew, kurtosis
import pandas as pd


def _mean_ci_normal(x, alpha=0.05):
    """Sample mean, variance, and a Normal-approximation CI for `x`."""
    x = np.asarray(x)
    n = x.size
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(var / n) if n > 0 else 0.0
    ci = np.array([mu - z * se, mu + z * se], dtype=float)
    return mu, var, ci


def simulate_heston_paths(model_params, S, r, delta, T, dt, n_paths, antithetic=True):
    """Full-truncation Euler simulation of Heston (S, v) paths.

    model_params: dict with kappa, theta, rho, sigma, v (initial variance).
    Returns (SPaths, vPaths), each shaped (n_paths, n_steps).
    """
    kappa = model_params["kappa"]
    theta = model_params["theta"]
    rho = model_params["rho"]
    sigma = model_params["sigma"]
    v0 = model_params["v"]

    n_steps = int(round(T / dt))
    SPaths = np.full((n_paths, n_steps), np.nan, dtype=float)
    vPaths = np.full((n_paths, n_steps), np.nan, dtype=float)

    err = None
    for j in range(n_paths):
        if not antithetic or j % 2 == 0:
            err = np.random.randn(2, n_steps)
        else:
            err = -err  # antithetic pair

        v = v0
        logS = np.log(S)
        for t in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            logS = logS + (r - delta - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * err[0, t]
            v = v + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos * dt) * (
                rho * err[0, t] + np.sqrt(1 - rho ** 2) * err[1, t]
            )
            SPaths[j, t] = np.exp(logS)
            vPaths[j, t] = v

    return SPaths, vPaths


def terminal_diagnostics_table(SPaths, vPaths):
    """Mean/median/skewness/kurtosis of log(S_T) and v_T, as a DataFrame."""
    s_T = np.log(SPaths[:, -1])
    v_T = vPaths[:, -1]
    tbl = np.array([
        [np.mean(s_T), np.median(s_T), skew(s_T, bias=False), kurtosis(s_T, fisher=False, bias=False)],
        [np.mean(v_T), np.median(v_T), skew(v_T, bias=False), kurtosis(v_T, fisher=False, bias=False)],
    ])
    return pd.DataFrame(tbl, index=["log(S_T)", "v_T"], columns=["mean", "median", "skewness", "kurtosis"])


def european_put_from_paths(SPaths, K, r, dt):
    """European put price/variance/CI from simulated terminal prices (antithetic-paired)."""
    n_steps = SPaths.shape[1]
    payoff = np.maximum(0.0, K - SPaths[:, -1]) * np.exp(-r * dt * n_steps)
    payoff_avg = 0.5 * (payoff[0::2] + payoff[1::2])
    return _mean_ci_normal(payoff_avg)


def _build_basis(order, SData, vData):
    """Regression basis for the LSM continuation-value fit, orders 1-5."""
    cols = [np.ones_like(SData), SData, SData ** 2]
    if order >= 2:
        cols.append(SData ** 3)
    if order == 5:
        cols.append(SData ** 4)
    cols += [vData, vData ** 2]
    if order >= 2:
        cols.append(vData ** 3)
    if order == 5:
        cols.append(vData ** 4)
    if order >= 3:
        cols.append(vData * SData)
    if order >= 4:
        cols += [vData ** 2 * SData, vData * SData ** 2]
    return np.column_stack(cols)


def lsm_american_put(SPaths, vPaths, K, r, dt, basis_orders=range(1, 6)):
    """American put price/CI via Longstaff-Schwartz, for each basis order.

    Returns (prices, cis) with prices shaped (len(basis_orders),) and cis
    shaped (2, len(basis_orders)).
    """
    n_paths, n_steps = SPaths.shape
    basis_orders = list(basis_orders)
    prices = np.zeros(len(basis_orders))
    cis = np.zeros((2, len(basis_orders)))

    for idx, order in enumerate(basis_orders):
        CashFlows = np.maximum(0.0, K - SPaths[:, -1]).copy()
        ExerciseIdx = np.full(n_paths, n_steps - 1, dtype=int)

        for step in range(n_steps - 2, -1, -1):
            InMoney = np.where(SPaths[:, step] < K)[0]
            if InMoney.size == 0:
                continue

            SData = SPaths[InMoney, step] / K
            vData = vPaths[InMoney, step]
            XData = _build_basis(order, SData, vData)

            df_step_to_ex = np.exp(-r * dt * (ExerciseIdx[InMoney] - step))
            YData = CashFlows[InMoney] * df_step_to_ex

            a, *_ = np.linalg.lstsq(XData, YData, rcond=None)
            ContinuationValue = XData @ a
            IntrinsicValue = K - SPaths[InMoney, step]

            Exercise = np.where(IntrinsicValue > ContinuationValue)[0]
            if Exercise.size > 0:
                k_idx = InMoney[Exercise]
                CashFlows[k_idx] = IntrinsicValue[Exercise]
                ExerciseIdx[k_idx] = step

        american_pv = CashFlows * np.exp(-r * dt * (ExerciseIdx + 1))
        american_pv_avg = 0.5 * (american_pv[0::2] + american_pv[1::2])
        mean, _, ci = _mean_ci_normal(american_pv_avg)
        prices[idx] = mean
        cis[:, idx] = ci

    return prices, cis


def price_lsm_heston(model_params, S, r, delta, K, T, dt, n_paths, basis_orders=range(1, 6)):
    """European + American (LSM) put prices under Heston, from one set of simulated paths."""
    SPaths, vPaths = simulate_heston_paths(model_params, S, r, delta, T, dt, n_paths)
    price_eur, _, ci_eur = european_put_from_paths(SPaths, K, r, dt)
    prices_amer, cis_amer = lsm_american_put(SPaths, vPaths, K, r, dt, basis_orders)
    return SPaths, vPaths, price_eur, ci_eur, prices_amer, cis_amer
