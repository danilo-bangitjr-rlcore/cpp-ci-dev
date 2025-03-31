from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp

import src.agent.components.networks.networks as nets


class GACNetParams(NamedTuple):
    critic_params: nets.Params
    actor_params: nets.Params
    proposal_params: nets.Params


class GACState(NamedTuple):
    params: GACNetParams


@dataclass
class GreedyACConfig:
    num_samples: int = 128
    actor_percentile: float = 0.1
    proposal_percentile: float = 0.1
    uniform_weight: float = 1.0


class CriticOutputs(NamedTuple):
    q: jax.Array


def critic_builder(cfg: nets.TorsoConfig):
    def _inner(x: jax.Array, a: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x, a)

        return CriticOutputs(
            q=hk.Linear(1)(phi),
        )

    return hk.without_apply_rng(hk.transform(_inner))


class ActorOutputs(NamedTuple):
    alpha: jax.Array
    beta: jax.Array


def actor_builder(cfg: nets.TorsoConfig, act_dim: int):
    def _inner(x: jax.Array):
        torso = nets.torso_builder(cfg)
        phi = torso(x)

        return ActorOutputs(
            alpha=hk.Linear(act_dim)(phi),
            beta=hk.Linear(act_dim)(phi),
        )

    return hk.without_apply_rng(hk.transform(_inner))


class GreedyAC:
    def __init__(self, cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int):
        self.seed = seed
        self.action_dim = action_dim
        self.num_samples = cfg.num_samples
        self.actor_percentile = cfg.actor_percentile
        self.proposal_percentile = cfg.proposal_percentile
        self.uniform_weight = cfg.uniform_weight

        critic_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LateFusionConfig(sizes=[32, 32], activation='relu'),
                nets.LinearConfig(size=64, activation='relu'),
            ],
        )
        self.critic = critic_builder(critic_torso_cfg)

        actor_torso_cfg = nets.TorsoConfig(
            layers=[
                nets.LinearConfig(size=64, activation='relu'),
                nets.LinearConfig(size=64, activation='relu'),
            ]
        )
        self.actor = actor_builder(actor_torso_cfg, action_dim)
        self.proposal = self.actor

        key = jax.random.PRNGKey(seed)
        params = GACNetParams(
            critic_params=self.critic.init(rng=key, x=jnp.zeros(state_dim), a=jnp.zeros(action_dim)),
            actor_params=self.actor.init(rng=key, x=jnp.zeros(state_dim)),
            proposal_params=self.proposal.init(rng=key, x=jnp.zeros(state_dim)),
        )
        self.agent_state = GACState(params)


    @partial(jax.jit, static_argnums=(0,))
    def get_actor_actions(self, params: GACNetParams, rng: chex.PRNGKey, states: jax.Array):
        actor_params = params.actor_params
        out: ActorOutputs = self.actor.apply(params=actor_params, x=states)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)


    @partial(jax.jit, static_argnums=(0,))
    def get_proposal_actions(self, params: GACNetParams, rng: chex.PRNGKey, states: jax.Array) -> jax.Array:
        proposal_params = params.proposal_params
        out: ActorOutputs = self.proposal.apply(params=proposal_params, x=states)
        return distrax.Beta(out.alpha, out.beta).sample(seed=rng)

    @partial(jax.jit, static_argnums=(0,))
    def get_uniform_actions(self, rng: chex.PRNGKey, samples: int) -> jax.Array:
        return jax.random.uniform(rng, (samples, self.action_dim))

    def critic_loss(self, params: GACNetParams):
        pass

    def actor_loss(self, params: GACNetParams):
        pass

    def proposal_loss(self, params: GACNetParams):
        pass

    def critic_update(self, params: GACNetParams):
        pass

    def actor_update(self, params: GACNetParams):
        pass

    def proposal_update(self, params: GACNetParams):
        pass

    def update(self, params: GACNetParams):
        pass
