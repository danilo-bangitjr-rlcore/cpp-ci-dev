import pandas as pd

from corerl.configs.environment.async_env import AsyncEnvConfig


class AsyncEnv:
    def emit_action(self, action: pd.DataFrame) -> None: ...

    def get_latest_obs(self) -> pd.DataFrame: ...

    def cleanup(self) -> None:
        return

    def get_cfg(self) -> AsyncEnvConfig: ...
