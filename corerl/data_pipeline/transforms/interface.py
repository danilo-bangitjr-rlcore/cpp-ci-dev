from dataclasses import dataclass

import pandas as pd


@dataclass
class TransformCarry:
    obs: pd.DataFrame
    transform_data: pd.DataFrame
    tag: str
