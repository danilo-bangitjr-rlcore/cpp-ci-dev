import numpy as np
import pandas as pd


class AsyncEnv:
    def emit_action(self, action: np.ndarray) -> None:
        ...

    def get_latest_obs(self) -> pd.DataFrame:
        ...
