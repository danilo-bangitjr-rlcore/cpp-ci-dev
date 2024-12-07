import pandas as pd
from dataclasses import dataclass

@dataclass
class TransformCarry:
    obs: pd.DataFrame
    transform_data: pd.DataFrame
    tag: str
