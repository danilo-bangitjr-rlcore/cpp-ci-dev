from dataclasses import dataclass

import jax

import src.agent.components.networks.networks as nets


@dataclass
class GACNetParams:
    critic_params: nets.Params
    actor_params: nets.Params
    proposal_params: nets.Params

@dataclass
class GreedyACConfig:
    num_samples: int = 128
    actor_percentile: float = 0.1
    proposal_percentile: float = 0.1
    uniform_weight: float = 1.0

class GreedyAC:
    def __init__(self, cfg: GreedyACConfig, seed: int, action_dim: int):
        self.seed = seed
        self.action_dim = action_dim
        self.num_samples = cfg.num_samples
        self.actor_percentile = cfg.actor_percentile
        self.proposal_percentile = cfg.proposal_percentile
        self.uniform_weight = cfg.uniform_weight

        self.critic_cfg = nets.EnsembleNetConfig(
            subnet=nets.FusionNetConfig(output_size=1),
            ensemble=1
        )
        actor_cfg = nets.LinearNetConfig(output_size=2*action_dim) # Two distribution parameters for Beta and Gaussian
        self.actor = nets.network_init(actor_cfg, input_dims=1)
        self.proposal = nets.network_init(actor_cfg, input_dims=1)

    @jax.jit
    def get_initial_params(self, sample_x: jax.Array) -> GACNetParams:
        critic_params = nets.ensemble_net_init(self.critic_cfg, self.seed, 2, sample_x)
        rng = jax.random.PRNGKey(self.seed)
        rngs = jax.random.split(rng, 2)
        actor_params = self.actor.init(rng=rngs[0], x=sample_x)
        proposal_params = self.proposal.init(rng=rngs[1], x=sample_x)

        return GACNetParams(critic_params, actor_params, proposal_params)

    @jax.jit
    def get_actor_actions(self, params: GACNetParams, states: jax.Array) -> jax.Array:
        actor_params = params.actor_params
        dist_params = self.actor.apply(params=actor_params, x=states)
        alphas = dist_params[:, 0::2]
        betas = dist_params[:, 1::2]
        rng = jax.random.PRNGKey(self.seed)

        return jax.random.beta(rng, alphas, betas)

    @jax.jit
    def get_proposal_actions(self, params: GACNetParams, states: jax.Array) -> jax.Array:
        proposal_params = params.proposal_params
        dist_params = self.proposal.apply(params=proposal_params, x=states)
        alphas = dist_params[:, 0::2]
        betas = dist_params[:, 1::2]
        rng = jax.random.PRNGKey(self.seed)

        return jax.random.beta(rng, alphas, betas)

    @jax.jit
    def get_uniform_actions(self, samples: int) -> jax.Array:
        rng = jax.random.PRNGKey(self.seed)
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
