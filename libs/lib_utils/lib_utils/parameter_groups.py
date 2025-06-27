from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import optax


@dataclass
class ParameterGroup:
    name: str
    parameter_keys: list[str]
    optimizer: optax.GradientTransformation | None = None
    opt_state: chex.ArrayTree | None = None


class ParameterGroupManager:
    def __init__(self):
        self._groups: dict[str, ParameterGroup] = {}

    def add_group(self, name: str, parameter_keys: list[str], optimizer: optax.GradientTransformation):
        self._groups[name] = ParameterGroup(
            name=name,
            parameter_keys=parameter_keys,
            optimizer=optimizer,
        )

    def get_group_gradients(self, grads: chex.ArrayTree, group_name: str) -> chex.ArrayTree:
        if group_name not in self._groups:
            return grads

        group = self._groups[group_name]

        def filter_grads(path: tuple) -> bool:
            path_str = '/'.join(str(p) for p in path)
            return any(key in path_str for key in group.parameter_keys)

        return jax.tree_util.tree_map_with_path(
            lambda path, value: value if filter_grads(path) else jnp.zeros_like(value),
            grads,
        )

    def init_optimizer_states(self, params: chex.ArrayTree) -> dict[str, chex.ArrayTree]:
        opt_states = {}
        for group_name, group in self._groups.items():
            if group.optimizer is not None:
                group.opt_state = group.optimizer.init(params)
                opt_states[group_name] = group.opt_state
        return opt_states

    def update_parameters(
        self,
        params: chex.ArrayTree,
        grads: chex.ArrayTree,
        opt_states: dict[str, chex.ArrayTree],
    ) -> tuple[chex.ArrayTree, dict[str, chex.ArrayTree]]:
        updated_params = params
        new_opt_states = {}

        for group_name, group in self._groups.items():
            if group.optimizer is not None and group_name in opt_states:
                group_grads = self.get_group_gradients(grads, group_name)
                current_opt_state = opt_states[group_name]

                updates, new_opt_state = group.optimizer.update(
                    group_grads, current_opt_state, params=params,
                )

                updated_params = optax.apply_updates(updated_params, updates)
                new_opt_states[group_name] = new_opt_state

        return updated_params, new_opt_states

    def has_groups(self) -> bool:
        return len(self._groups) > 0
