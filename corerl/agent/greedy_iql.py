from omegaconf import DictConfig

from corerl.agent.iql import IQL
from corerl.agent.greedy_ac import GreedyAC
from pathlib import Path
import pickle as pkl


class GreedyIQL(GreedyAC, IQL):
    """
    A verison of IQL that uses GAC-style updates.
    """
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)

    def update_actor(self):
        GreedyAC.update_actor(self)

    def update_critic(self):
        IQL.update_critic(self)

    def update(self, share_batch: bool = True) -> None:
        GreedyAC.update(self)

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

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pkl.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_path = path / "actor"
        self.actor.load(actor_path)

        sampler_path = path / "sampler"
        self.sampler.load(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        v_critic_path = path / "v_critic"
        self.v_critic.load(v_critic_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pkl.load(f)
