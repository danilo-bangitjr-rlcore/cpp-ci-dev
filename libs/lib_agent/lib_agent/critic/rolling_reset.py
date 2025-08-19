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
    is_warmed_up: bool = False
    is_active: bool = True

@dataclass
class RollingResetManagerStatus:
    total_critics: int
    active_critics: int
    active_indices: list[int]
    metadata: dict[int, CriticInfo]

class RollingResetManager:
    def __init__(self, config: RollingResetConfig, ensemble_size: int):
        self._config = config
        self._update_count = 0

        self._total_critics = ensemble_size
        self._critic_info = [CriticInfo() for _ in range(self._total_critics)]
        self._active_indices = set(range(ensemble_size))

    @property
    def total_critics(self) -> int:
        return self._total_critics

    @property
    def active_indices(self) -> set[int]:
        return self._active_indices.copy()

    def should_reset(self) -> bool:
        return self._update_count % self._config.reset_period == 0

    def increment_update_count(self):
        self._update_count += 1

    def get_critic_metrics(self, critic_idx: int, prefix: str = "") -> dict[str, float]:
        info = self._critic_info[critic_idx]
        metrics = {
            f"CRITIC{critic_idx}_is_active": float(info.is_active),
            f"CRITIC{critic_idx}_is_warmed_up": float(info.is_warmed_up),
            f"CRITIC{critic_idx}_birthdate": info.birthdate,
            f"CRITIC{critic_idx}_training_steps": info.training_steps,
            f"CRITIC{critic_idx}_recent_loss": info.recent_loss,
        }

        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        return metrics

    def update_critic_metadata(self, losses: jax.Array):
        for i in range(self._total_critics):
            loss_value = losses[i]
            if loss_value.ndim > 0:
                loss_value = loss_value.mean()
            self._critic_info[i].recent_loss = float(loss_value)
            self._critic_info[i].training_steps += 1

            if (self._critic_info[i].training_steps >= self._config.warm_up_steps and
                not self._critic_info[i].is_warmed_up):
                self._critic_info[i].is_warmed_up = True
                if i not in self._active_indices:
                    self._active_indices.add(i)
                    self._critic_info[i].is_active = True

    def _get_critic_score(self, critic_idx: int) -> float:
        return float(self._critic_info[critic_idx].birthdate)

    def _select_critic_for_reset(self) -> int | None:
        active_critics = list(self._active_indices)

        if not active_critics:
            return None

        warmed_up_critics = [
            idx for idx in active_critics
            if self._critic_info[idx].is_warmed_up
        ]

        if not warmed_up_critics:
            return None

        return min(warmed_up_critics, key=self._get_critic_score)


    def _select_background_critic(self) -> int | None:
        ready_background_critics = [
            i for i in range(self._total_critics)
            if (i not in self._active_indices and
                self._critic_info[i].training_steps >= self._config.warm_up_steps)
        ]

        if not ready_background_critics:
            return None

        return min(ready_background_critics, key=lambda x: self._critic_info[x].birthdate)

    def reset(
        self,
        critic_state: CriticState,
        rng: jax.Array,
        init_member_fn: Callable[[jax.Array, jax.Array, jax.Array], CriticState],
        state_dim: int,
        action_dim: int,
    ) -> CriticState:
        critic_to_reset = self._select_critic_for_reset()
        if critic_to_reset is None:
            return critic_state


        # Reset critic state
        self._critic_info[critic_to_reset].training_steps = 0
        self._critic_info[critic_to_reset].birthdate = self._update_count
        self._critic_info[critic_to_reset].recent_loss = float('inf')
        self._critic_info[critic_to_reset].is_warmed_up = False
        self._critic_info[critic_to_reset].is_active = False

        # remove from active set
        self._active_indices.discard(critic_to_reset)

        selected_background_critic = self._select_background_critic()
        if selected_background_critic is None:
            return critic_state

        self._active_indices.remove(critic_to_reset)
        self._active_indices.add(selected_background_critic)

        # initialize new member state
        x_dummy = jnp.zeros(state_dim)
        a_dummy = jnp.zeros(action_dim)
        new_member_state = init_member_fn(rng, x_dummy, a_dummy)

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

    def get_status(self) -> RollingResetManagerStatus:
        return RollingResetManagerStatus(
            total_critics=self._total_critics,
            active_critics=len(self._active_indices),
            active_indices=sorted(self._active_indices),
            metadata={k: self._critic_info[k] for k in range(self._total_critics)},
        )
