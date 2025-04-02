from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax import Array, random, vmap

from src.interaction.transition_creator import Transition


class VectorizedTransition(NamedTuple):
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    done: jax.Array
    gamma: jax.Array

class EnsembleReplayBuffer:
    def __init__(self, n_ensemble: int = 2, max_size: int = 1_000_000,
                 ensemble_prob: float = 0.5, batch_size:int = 256, seed: int = 0):
        self.max_size = max_size
        self.n_ensemble = n_ensemble
        self.ensemble_prob = ensemble_prob
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size

        self.transitions = None #: List[Transition | None] = [None] * max_size
        self.ensemble_masks = jnp.zeros((n_ensemble, max_size), dtype=bool)
        self.key = random.PRNGKey(seed)

    def _init_transitions(self, transition: Transition) -> None:
        self.transitions = VectorizedTransition(
            state=jnp.zeros((self.max_size, transition.state_dim)),
            action=jnp.zeros((self.max_size,transition.action_dim)),
            reward=jnp.zeros((self.max_size, 1)),
            next_state=jnp.zeros((self.max_size, transition.state_dim)),
            done=jnp.zeros((self.max_size, 1)),
            gamma=jnp.zeros((self.max_size, 1)),
        )

    def _add_transition(self, transition: Transition):
        assert self.transitions is not None
        # Create updated transitions with new data at the current pointer
        self.transitions = VectorizedTransition(
            state=self.transitions.state.at[self.ptr].set(transition.prior.state),
            action=self.transitions.action.at[self.ptr].set(transition.post.action),
            reward=self.transitions.reward.at[self.ptr].set(transition.n_step_reward),
            next_state=self.transitions.next_state.at[self.ptr].set(transition.post.state),
            done=self.transitions.done.at[self.ptr].set(transition.post.done),
            gamma=self.transitions.gamma.at[self.ptr].set(transition.post.gamma),
        )

    def add(self, transition: Transition) -> None:
        if self.transitions is None:
            self._init_transitions(transition)

        self._add_transition(transition)

        self.key, mask_key = random.split(self.key)
        ensemble_mask = random.uniform(mask_key, (self.n_ensemble,)) < self.ensemble_prob

        if not jnp.any(ensemble_mask):
            self.key, member_key = random.split(self.key)
            random_member = random.randint(member_key, (), 0, self.n_ensemble)
            ensemble_mask = ensemble_mask.at[random_member].set(True)

        self.ensemble_masks = self.ensemble_masks.at[:, self.ptr].set(ensemble_mask)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @partial(jax.jit, static_argnums=(0,))
    def sample(self) -> VectorizedTransition:
        assert self.transitions is not None

        ensemble_keys = random.split(self.key, self.n_ensemble + 1) # Create keys for each ensemble member
        self.key = ensemble_keys[0]  # Update the main key
        ensemble_keys = ensemble_keys[1:]  # Keys for sampling

        all_indices = jnp.arange(self.size)
        def vmap_sample(key: chex.PRNGKey, mask: Array):
            # Sample with weighted probabilities
            weights = jnp.where(
                mask > 0,
                mask / jnp.sum(mask),
                0,
            )
            return random.choice(
                key,
                all_indices,
                shape=(self.batch_size,),
                replace=True,
                p=weights
            )

        random_indices =  jax.vmap(vmap_sample)(ensemble_keys, self.ensemble_masks[:, :self.size])

        def index(indices: Array):
            return jax.tree_map(
                lambda x: x[indices],
                self.transitions
            )
        v_mapped_index = vmap(index)
        return v_mapped_index(random_indices)
