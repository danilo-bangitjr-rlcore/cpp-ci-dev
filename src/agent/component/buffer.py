from typing import NamedTuple, List
import jax
import jax.numpy as jnp
from jax import random

class Transition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    done: jnp.ndarray

class EnsembleReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, n_ensemble: int = 2, 
                 max_size: int = 1_000_000, ensemble_prob: float = 0.5, seed: int = 0):
        self.max_size = max_size
        self.n_ensemble = n_ensemble
        self.ensemble_prob = ensemble_prob
        self.ptr = 0
        self.size = 0
        
        self.state = jnp.zeros((max_size, state_dim))
        self.action = jnp.zeros((max_size, action_dim))
        self.reward = jnp.zeros((max_size, 1))
        self.next_state = jnp.zeros((max_size, state_dim))
        self.done = jnp.zeros((max_size, 1))
        
        self.ensemble_masks = jnp.zeros((n_ensemble, max_size), dtype=bool)
        self.key = random.PRNGKey(seed)
    
    def add(self, state: jnp.ndarray, action: jnp.ndarray, reward: float, 
            next_state: jnp.ndarray, done: bool) -> None:
        self.state = self.state.at[self.ptr].set(state)
        self.action = self.action.at[self.ptr].set(action)
        self.reward = self.reward.at[self.ptr].set(reward)
        self.next_state = self.next_state.at[self.ptr].set(next_state)
        self.done = self.done.at[self.ptr].set(done)
        
        self.key, mask_key = random.split(self.key)
        ensemble_mask = random.uniform(mask_key, (self.n_ensemble,)) < self.ensemble_prob
        
        if not jnp.any(ensemble_mask):
            self.key, member_key = random.split(self.key)
            random_member = random.randint(member_key, (), 0, self.n_ensemble)
            ensemble_mask = ensemble_mask.at[random_member].set(True)
            
        self.ensemble_masks = self.ensemble_masks.at[:, self.ptr].set(ensemble_mask)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> List[Transition]:
        ensemble_batches = []
        
        for i in range(self.n_ensemble):
            valid_indices = jnp.where(self.ensemble_masks[i, :self.size])[0]
            
            self.key, subkey = random.split(self.key)
            ind = random.choice(subkey, valid_indices, (batch_size,))
            
            batch = Transition(
                state=self.state[ind],
                action=self.action[ind],
                reward=self.reward[ind],
                next_state=self.next_state[ind],
                done=self.done[ind]
            )
            ensemble_batches.append(batch)
        
        return ensemble_batches 