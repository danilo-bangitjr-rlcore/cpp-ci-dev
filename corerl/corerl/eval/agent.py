from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec, Protocol, SupportsFloat, TypeVar

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_config.config import config
from pydantic import BaseModel

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import GreedyAC
from corerl.component.buffer import JaxTransition
from corerl.state import AppState

# --------------------------------- Utilities --------------------------------- #

class EvalConfig(Protocol):
    enabled: bool

T = TypeVar('T', bound=EvalConfig)
P = ParamSpec('P')
EvalFn = Callable[Concatenate[T, GreedyAC, P], Sequence[jax.Array]]

def agent_eval(
    app_state: AppState,
    agent: BaseAgent,
    cfg_lens : Callable[[AppState], T],
    eval_fn: EvalFn[T, P],
    metric_names: list[str],
    *args: P.args,
    **kwargs: P.kwargs,
):
    cfg = cfg_lens(app_state)
    if not cfg.enabled:
        return

    if not isinstance(agent, GreedyAC):
        return

    metrics_tuple = eval_fn(cfg, agent, *args, **kwargs)

    if len(metrics_tuple) == 0:
        return

    for name, values in zip(metric_names, metrics_tuple, strict=True):
        metrics: dict[str, SupportsFloat] = {}
        if values.size == 1:
            metrics[name] = values.max()
        else:
            metrics = {
                f'{name}_max':  values.max(),
                f'{name}_min':  values.min(),
                f'{name}_mean': values.mean(),
                f'{name}_var':  values.var(),
            }

        for k, v in metrics.items():
            app_state.metrics.write(app_state.agent_step, k, v)

SAEvalFn = Callable[[T, GreedyAC, jax.Array, jax.Array, jax.Array], Sequence[jax.Array]]
BatchSAEvalFn = Callable[[EvalConfig, GreedyAC], Sequence[jax.Array]]
def policy_buffer_batchify(eval_fn: SAEvalFn) ->  BatchSAEvalFn:
    def batchified(cfg: EvalConfig, agent: GreedyAC):
        if not agent.is_policy_buffer_sampleable:
            return ()

        batches = agent.sample_policy_buffer()
        assert len(batches) == 1
        batch: JaxTransition = jax.tree.map(lambda x: x[0], batches)
        state_batch = batch.state
        action_lo_batch = batch.action_lo
        action_hi_batch = batch.action_hi

        return eval_fn(cfg, agent, state_batch, action_lo_batch, action_hi_batch)

    return batchified


# ------------------------------ Q Values Online ----------------------------- #

@config()
class QOnlineConfig:
    enabled: bool = True

def _q_online(
    cfg: QOnlineConfig,
    agent: GreedyAC,
    state: np.ndarray | jax.Array,
    action: np.ndarray | jax.Array,
):
    """
    Records the Q value of the action taken, the variance across the ensemble
    and the individual Q values for the ensemble members.
    """
    assert state.ndim == 1
    assert action.ndim == 1
    out = agent.get_values(
        jnp.expand_dims(state, [0, 1]),
        jnp.expand_dims(action, [0, 1]),
    )

    return out.reduced_value, out.ensemble_values, out.ensemble_variance

def q_online(
        app_state: AppState,
        agent: BaseAgent,
        state: np.ndarray | jax.Array,
        action: np.ndarray | jax.Array,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.q_online,
        eval_fn=_q_online,
        metric_names=['q', 'q_ensemble', 'q_ensemble_variance'],
        state=state,
        action=action,
    )

# ------------------------------ Greed Dist ------------------------------ #

def get_max_action(actions: np.ndarray | jax.Array, values: jax.Array):
    max_indices = jnp.argmax(values, axis=0)
    return actions[max_indices, :]

@config()
class GreedDistConfig:
    enabled: bool = True
    n_samples: int = 100

def _greed_dist(
    cfg: GreedDistConfig,
    agent: GreedyAC,
    states: np.ndarray | jax.Array,
    action_lo: np.ndarray | jax.Array,
    action_hi: np.ndarray | jax.Array,
) -> Sequence[jax.Array]:
    """
    Evaluates whether the policy is greedy w.r.t. the critic in terms of a distance metric.

    For some state in the policy buffer, we sample n_samples uniformly at random.
    The greedification for this state is the l2 distance between the action that
    maximizes the Q value and the action that maximizes the policy.

    Returns the metric for each state in the batch.
    """
    assert states.ndim == action_lo.ndim == action_hi.ndim == 1

    uniform_actions = np.random.uniform(
        low=action_lo,
        high=action_hi,
        size=(cfg.n_samples, agent.action_dim),
    )

    q_values = agent.get_values(
        # add (ensemble, batch) dimensions
        jnp.expand_dims(states, [0, 1]),
        jnp.expand_dims(uniform_actions, [0, 1]),
    )

    q = q_values.reduced_value.squeeze(0)
    chex.assert_shape(q, (cfg.n_samples, 1))

    max_actions_critic = get_max_action(uniform_actions, q)
    chex.assert_shape(max_actions_critic, (1, agent.action_dim))

    # Get log probabilities for the sampled actions from the actor.
    probs = agent.prob(
        # add batch dimension
        jnp.expand_dims(states, 0),
        jnp.expand_dims(uniform_actions, 0),
    ).squeeze(0)
    chex.assert_shape(probs, (cfg.n_samples,))

    # Get the max direct action according to log_probs for each state
    max_actions_actor = get_max_action(uniform_actions, jnp.expand_dims(probs, 1))
    chex.assert_shape(max_actions_actor, (1, agent.action_dim))

    # Calculate the L2 distance between the actions that maximize q values and the actions that maximize policy
    diff_actions = max_actions_critic - max_actions_actor
    distances = jnp.linalg.norm(diff_actions, axis=1)

    return (distances,)

def greed_dist_online(
        app_state: AppState,
        agent: BaseAgent,
        states: np.ndarray | jax.Array,
        action_lo: np.ndarray | jax.Array,
        action_hi: np.ndarray | jax.Array,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_dist_online,
        eval_fn=_greed_dist,
        metric_names=['greed_dist_online'],
        states=states,
        action_lo=action_lo,
        action_hi=action_hi,
    )


def greed_dist_batch(app_state: AppState, agent: BaseAgent):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_dist_batch,
        eval_fn=policy_buffer_batchify(_greed_dist),
        metric_names=['greed_dist_batch'],
    )

# ------------------------------ Q and PDF Plots ----------------------------- #

@config()
class QPDFPlotsConfig:
    enabled: bool = True
    # number of samples for the primary action (i.e. the values on the x-axis)
    primary_action_samples: int = 101
    # number of samples for other actions (i.e. how many times to average to create each point on the y-axis)
    other_action_samples: int = 10


def q_values_and_act_prob(
    app_state: AppState,
    agent: GreedyAC,
    state: np.ndarray | jax.Array,
    action_lo: np.ndarray | jax.Array,
    action_hi: np.ndarray | jax.Array,
):
    """
    Logs the probability density function of the policy and the Q values.
    This entries are of the form (metric, x, y) where
    * metric tells us the action being varied (think x-axis)
    * x is a direct action
    * y is the probability or Q value, averaged over samples where the action in metric is set to x
        and the other actions are sampled from the policy. cfg.other_action_samples controls the number of samples
    """
    cfg = app_state.cfg.eval_cfgs.q_pdf_plots
    if not cfg.enabled:
        return

    state = jnp.asarray(state)
    action_lo = jnp.asarray(action_lo)
    action_hi = jnp.asarray(action_hi)
    chex.assert_rank((state, action_lo, action_hi), 1)

    rng = jax.random.PRNGKey(app_state.agent_step)
    dist = agent.get_dist(state)

    on_policy_actions: jax.Array = dist.sample(
        seed=rng,
        sample_shape=(cfg.primary_action_samples, cfg.other_action_samples),
    )
    chex.assert_shape(on_policy_actions, (cfg.primary_action_samples, cfg.other_action_samples, agent.action_dim))

    x_axis_actions = jnp.linspace(0, 1, cfg.primary_action_samples)

    for a_dim_idx in range(agent.action_dim):
        query_actions = on_policy_actions.at[:, :, a_dim_idx].set(jnp.expand_dims(x_axis_actions, 1))
        probs = dist.prob(query_actions)
        chex.assert_shape(probs, (cfg.primary_action_samples, cfg.other_action_samples))

        measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(x_axis_actions, np.asarray(probs.mean(axis=1)), strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"pdf_plot_action_{a_dim_idx}",
            value=measure.model_dump_json(),
        )

        # Next, plot q values for the entire range of direct actions
        # need to loop over the "other_action" dimension
        # also need (ens, batch, n_samples) dimensions
        q_func = jax_u.vmap(agent.critic.get_values, (None, None, 3))
        qs = q_func(
            agent._critic_state.params,
            jnp.expand_dims(state, [0, 1]),
            jnp.expand_dims(query_actions, [0, 1]),
        )

        # because we looped the other_action dimension first,
        # we need some reordering
        qs = qs.transpose(1, 2, 3, 0, 4)
        chex.assert_shape(
            qs,
            (agent.ensemble, 1, cfg.primary_action_samples, cfg.other_action_samples, 1),
        )

        # avg over ensemble, batch, and other_action dimensions, remove the trailing value dim
        qs = qs.mean(axis=(0, 1, 3)).squeeze(-1)

        measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(x_axis_actions, qs, strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"qs_plot_action_{a_dim_idx}",
            value=measure.model_dump_json(),
        )


class XY(BaseModel):
    x: float
    y: float

class XYEval(BaseModel):
    data: list[XY]
