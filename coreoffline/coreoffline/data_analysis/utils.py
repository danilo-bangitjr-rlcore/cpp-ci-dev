import pandas as pd


def get_tags(data: list[pd.DataFrame]) -> list[str]:
    return list(data[0].columns)
