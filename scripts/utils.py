"""Utility helpers for notebook refactor.

Functions:
- split_by_cp(df, min_rows=10, max_rows=200) -> list of DataFrames
"""

import pandas as pd


def split_by_cp(df: pd.DataFrame, min_rows: int = 10, max_rows: int = 200):
    """Return a list of DataFrames, one per 担当CP, filtered by row counts.

    Keeps CPs with min_rows <= rows < max_rows.
    """
    result = []
    if '担当CP' not in df.columns:
        return result

    unique_cps = df['担当CP'].dropna().unique()
    for cp in unique_cps:
        df_cp = df[df['担当CP'] == cp].copy()
        if len(df_cp) >= min_rows and len(df_cp) < max_rows:
            result.append(df_cp)
    return result
