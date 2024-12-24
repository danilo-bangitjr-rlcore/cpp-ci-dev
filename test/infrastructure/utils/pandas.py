import numpy as np
import pandas as pd


def dfs_close(df1: pd.DataFrame, df2: pd.DataFrame, col_order_matters=False):
    if col_order_matters:
        if not df1.columns.equals(df2.columns):
            return False
    else:
        if set(df1.columns) != set(df2.columns):
            return False

    for col in df1.columns:
        if not np.allclose(df1[col], df2[col], equal_nan=True):
            return False

    return True
