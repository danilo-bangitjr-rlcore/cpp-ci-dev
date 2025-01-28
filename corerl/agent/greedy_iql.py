import pickle as pkl
from pathlib import Path
from typing import Literal

from pydantic import Field

from corerl.agent.base import BaseAgentConfig
from corerl.agent.greedy_ac import GreedyAC, GreedyACConfig
from corerl.agent.iql import IQL, IQLConfig
from corerl.configs.config import config
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState


@config(frozen=True)
class GreedyIQLConfig(BaseAgentConfig):
    name: Literal['greedy_iql'] = 'greedy_iql'

    gac: GreedyACConfig = Field(default_factory=GreedyACConfig)
    iql: IQLConfig = Field(default_factory=IQLConfig)

    temp: float = 1.0
    expectile: float = 0.8


class GreedyIQL(GreedyAC, IQL):
    """
    A verison of IQL that uses GAC-style updates.
    """
    def __init__(self, cfg: GreedyIQLConfig, app_state: AppState, col_desc: ColumnDescriptions):
        GreedyAC.__init__(self, cfg.gac, app_state, col_desc)
        IQL.__init__(self, cfg.iql, app_state, col_desc)

    def update_actor(self):
        return GreedyAC.update_actor(self)

    def update_critic(self) -> list[float]:
        return IQL.update_critic(self)

    def update(self) -> list[float]:
        return GreedyAC.update(self)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        actor_path = path / "actor"
        self.actor.save(actor_path)

        sampler_path = path / "sampler"
        self.sampler.save(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        v_critic_path = path / "v_critic"
        self.v_critic.save(v_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "wb") as f:
            pkl.dump(self.policy_buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        sampler_path = path / "sampler"
        self.sampler.load(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "rb") as f:
            self.policy_buffer = pkl.load(f)
