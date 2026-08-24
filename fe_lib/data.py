"""Excel data loaders. All paths are relative to fe_lib.paths.DATA_DIR."""

import numpy as np
import pandas as pd

from fe_lib.paths import DATA_DIR


def load_spx_returns(filename="SPX.xlsx"):
    """Dates and SPX log-returns from the daily price series (Lecture 1)."""
    df = pd.read_excel(DATA_DIR / filename, sheet_name=0, usecols="A:B", skiprows=5, nrows=8968)
    dates = pd.to_datetime(df.iloc[:, 0])
    SPX = df.iloc[:, 1].values
    SPXreturns = np.diff(np.log(SPX))
    return dates[1:], SPX, SPXreturns


def load_single_asset_options(filename="Assignment_1.xlsx", sheet="TSLA"):
    """Flat option-chain table (columns: strike, type, price) for one underlying."""
    return pd.read_excel(DATA_DIR / filename, sheet_name=sheet)


def load_option_grid(filename, sheet="options"):
    """Maturity x strike option grid, in the layout used by *_options.xlsx:
    row 0 = maturities (days), rows 2-14 = strikes, rows 16-28 = prices, columns 1-5.

    Returns (maturity_years, strike, price), each a numpy array.
    """
    df = pd.read_excel(DATA_DIR / filename, sheet_name=sheet)
    maturity = np.array(df.iloc[0, 1:6], dtype=np.float64) / 365.0
    strike = np.array(df.iloc[2:15, 1:6], dtype=np.float64)
    price = np.array(df.iloc[16:29, 1:6], dtype=np.float64)
    return maturity, strike, price
