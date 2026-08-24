# Financial Engineering — CBS

Teaching materials for the Financial Engineering course (MSc level) at
Copenhagen Business School.

**Prerequisites:** Black-Scholes, continuous-time stochastic processes, Ito's Lemma.

## Contents

| Path | Description |
|---|---|
| `CBS_FE.ipynb` | Teaching notebook: Lectures 1–5 and the associated in-class assignments. Shared with students from the start of the course. |
| `CBS_FE_Mandatory_Assignment.ipynb` | Mandatory assignment (volatility surface, Heston calibration, LSM pricing). Shared with students only *after* the hand-in deadline, as the reference solution. |
| `fe_lib/` | Shared Python library backing both notebooks (pricing, calibration, simulation). |
| `Data/` | Excel input files (SPX time series, option chains) used by the notebooks. |

## `fe_lib` overview

| Module | Covers |
|---|---|
| `black_scholes.py` | BSM pricing, greeks, implied volatility |
| `characteristic_functions.py` | Characteristic functions for BSM, Heston (SV), jump-to-ruin, SVJ, Kou |
| `integration.py` | Gauss-Legendre / trapezoidal integration for Fourier-inversion pricing |
| `pricing.py` | Fourier-inversion pricing, implied vols, calibration residuals |
| `local_vol.py` | Risk-neutral density (Breeden-Litzenberger) and Dupire local volatility |
| `sabr.py` | SABR implied-volatility approximation |
| `simulation.py` | GBM path simulation, Monte Carlo variance reduction |
| `lsm.py` | Heston path simulation and Longstaff-Schwartz American option pricing |
| `data.py` | Excel loaders for the files in `Data/` |
| `paths.py` | Central definition of `DATA_DIR` / output directory |
| `plotting.py` | Shared save/show helper for figures |

Each module carries a one-line docstring stating its purpose, and each function
a short docstring/comment where the *why* isn't obvious from the code — no
verbose docstrings, since the audience is expected to read the math alongside
the code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib openpyxl jupyter ipykernel
```

Run notebooks from the repository root so relative imports (`from fe_lib...`)
and data paths resolve correctly.
