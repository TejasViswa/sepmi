# sepmi/io.py
import numpy as np
import pandas as pd


def load_series_any(path):
    """
    Load a pandas Series from CSV or pickle. For CSV, chooses a likely column
    if a DataFrame is provided.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        if isinstance(df, pd.DataFrame):
            if df.shape[1] == 1:
                return df.iloc[:, 0]
            for c in [
                "price","prices","close","Close","PRC","VALUE",
                "weight","weights","w","benchmark_wt"
            ]:
                if c in df.columns:
                    return df[c]
            return df.iloc[:, 0]
        return pd.Series(df.squeeze())
    else:
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        raise TypeError(f"Unsupported object in {path}: {type(obj)}")


def align_prices_weights(prices: pd.Series, weights: pd.Series):
    """
    Align price and weight Series on common index and normalize weights to sum 1.
    """
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    weights = pd.to_numeric(weights, errors="coerce").dropna()
    common = prices.index.intersection(weights.index)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between prices and target weights.")
    prices = prices.loc[common].astype(float)
    weights = weights.loc[common].astype(float)
    s = weights.sum()
    if not np.isfinite(s) or abs(s) < 1e-16:
        raise ValueError("Target weights sum is zero or invalid.")
    if abs(s - 1.0) > 1e-6:
        weights = weights / s
    return prices, weights


def align_covariance_and_target(cov, tgt):
    """
    Ensure covariance is a square labeled DataFrame, align target Series to it,
    and lightly symmetrize/jitter the covariance for numerical niceness.
    """
    if isinstance(cov, np.ndarray):
        cov = pd.DataFrame(cov)
    if isinstance(tgt, pd.DataFrame):
        tgt = tgt.iloc[:, 0]
    if not isinstance(cov, pd.DataFrame) or not isinstance(tgt, pd.Series):
        raise TypeError("Expected cov: DataFrame (or ndarray), tgt: Series (or 1-col DataFrame).")

    if not cov.index.equals(cov.columns):
        if len(cov.index) != len(cov.columns):
            n = cov.shape[0]
            cov.index = [f"ASSET_{i}" for i in range(n)]
            cov.columns = cov.index

    common = cov.index.intersection(tgt.index)
    if len(common) == 0:
        if len(tgt) == cov.shape[0]:
            tgt.index = cov.index
            common = cov.index
        else:
            raise ValueError("No overlapping tickers between covariance and target.")
    cov = cov.loc[common, common].copy()
    tgt = tgt.loc[common].astype(float).copy()

    M = cov.values.astype(float, copy=True)
    M = 0.5 * (M + M.T)
    eps = 1e-10 * np.trace(M) / max(1, M.shape[0])
    np.fill_diagonal(M, np.diag(M) + eps)
    cov.loc[:, :] = M
    return cov, tgt 