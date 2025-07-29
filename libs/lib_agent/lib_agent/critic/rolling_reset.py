from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from lib_agent.critic.critic_utils import CriticState, RollingResetConfig


@dataclass
class CriticInfo:
    birthdate: int = 0
    training_steps: int = 0
    recent_loss: float = float('inf')

@dataclass
class RollingResetManagerStatus:
    total_critics: int
    active_critics: int
    background_critics: int
    active_indices: list[int]
    background_indices: list[int]
    metadata: dict[int, CriticInfo]
    next_reset_idx: int

class RollingResetManager:
    def __init__(self, config: RollingResetConfig, ensemble_size: int):
        self._config = config
        self._ensemble_size = ensemble_size
        self._update_count = 0
        self._next_reset_idx = 0

        self._critic_info = [CriticInfo() for _ in range(ensemble_size + config.num_background_critics)]
        self._total_critics = ensemble_size + config.num_background_critics
        self._active_indices = set(range(ensemble_size))

    @property
    def total_critics(self) -> int:
        return self._total_critics

    @property
    def active_indices(self) -> set[int]:
        return self._active_indices.copy()

    def should_reset(self) -> bool:
        if self._config.num_background_critics == 0:
            return self._update_count % self._config.reset_period == 0
        return False

    def increment_update_count(self):
        self._update_count += 1

    def update_critic_metadata(self, losses: jax.Array):
        for i in range(self._total_critics):
            loss_value = losses[i]
            if loss_value.ndim > 0:
                loss_value = loss_value.mean()
            self._critic_info[i].recent_loss = float(loss_value)
            self._critic_info[i].training_steps += 1

    def _select_critics_for_reset(self) -> int | None:
        if self._config.num_background_critics == 0:
            return None

        active_critics_list = sorted(self._active_indices)
        return active_critics_list[self._next_reset_idx % len(active_critics_list)]

    def _select_background_critic(self) -> int | None:
        ready_background_critics = [
            i for i in range(self._total_critics)
            if (i not in self._active_indices and
                self._critic_info[i].training_steps >= self._config.background_training_steps)
        ]

        if not ready_background_critics:
            return None

        # sort by birthdate
        sorted_birthdates = sorted(
            ready_background_critics,
            key=lambda x: self._critic_info[x].birthdate,
        )
        return sorted_birthdates[0]

    def reset(
        self,
        critic_state: CriticState,
        rng: jax.Array,
        init_member_fn: Callable[[jax.Array, jax.Array, jax.Array], CriticState],
        state_dim: int,
        action_dim: int,
    ) -> CriticState:
        critic_to_reset = self._select_critics_for_reset()

        if critic_to_reset is None:
            return critic_state

        selected_background_critic = self._select_background_critic()

        if selected_background_critic is None:
            return critic_state

        self._active_indices.remove(critic_to_reset)
        self._active_indices.add(selected_background_critic)

        self._critic_info[critic_to_reset].training_steps = 0
        self._critic_info[critic_to_reset].birthdate = self._update_count
        self._critic_info[critic_to_reset].recent_loss = float('inf')

        self._critic_info[selected_background_critic].birthdate = self._update_count

        self._next_reset_idx += 1

        rng, init_rng = jax.random.split(rng)
        x_dummy = jnp.zeros(state_dim)
        a_dummy = jnp.zeros(action_dim)
        new_member_state = init_member_fn(init_rng, x_dummy, a_dummy)

        return self._apply_reset_to_state(critic_state, critic_to_reset, new_member_state)

    def _apply_reset_to_state(
        self,
        critic_state: CriticState,
        critic_to_reset: int,
        new_member_state: CriticState,
    ) -> CriticState:
        new_params = jax.tree.map(
            lambda ensemble_param, new_param: ensemble_param.at[critic_to_reset].set(new_param),
            critic_state.params, new_member_state.params,
        )
        new_opt_state = jax.tree.map(
            lambda ensemble_opt, new_opt: ensemble_opt.at[critic_to_reset].set(new_opt),
            critic_state.opt_state, new_member_state.opt_state,
        )

        return CriticState(params=new_params, opt_state=new_opt_state)

    @property
    def background_indices(self) -> list[int]:
        return sorted(set(range(self._total_critics)) - self._active_indices)

    def get_status(self) -> RollingResetManagerStatus:
        return RollingResetManagerStatus(
            total_critics=self._total_critics,
            active_critics=len(self._active_indices),
            background_critics=len(self.background_indices),
            active_indices=sorted(self._active_indices),
            background_indices=self.background_indices,
            metadata={k: self._critic_info[k] for k in range(self._total_critics)},
            next_reset_idx=self._next_reset_idx,
        )
