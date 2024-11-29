import pandas as pd
from dataclasses import dataclass

@dataclass
class TransformCarry:
    obs: pd.DataFrame
    agent_state: pd.DataFrame
    tag: str
