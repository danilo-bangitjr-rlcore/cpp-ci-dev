from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec, Protocol, SupportsFloat, TypeVar

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.buffer.datatypes import JaxTransition
from lib_utils.errors import fail_gracefully
from pydantic import BaseModel

from corerl.agent.base import BaseAgent
from corerl.agent.greedy_ac import GreedyAC
from corerl.configs.eval.agent import GreedDistConfig, QOnlineConfig
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

def _q_online(
    cfg: QOnlineConfig,
    agent: GreedyAC,
    state: jax.Array,
    action: jax.Array,
):
    """
    Records the Q value of the action taken, the variance across the ensemble
    and the individual Q values for the ensemble members.
    """
    assert state.ndim == 1
    assert action.ndim == 1
    out = agent.get_active_values(state, action)

    return out.reduced_value, out.ensemble_values, out.ensemble_variance

def q_online(
        app_state: AppState,
        agent: BaseAgent,
        state: jax.Array,
        action: jax.Array,
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

def get_max_action(actions: jax.Array, values: jax.Array):
    max_indices = jnp.argmax(values, axis=0)
    return actions[max_indices, :]

def _greed_dist(
    cfg: GreedDistConfig,
    agent: GreedyAC,
    state: jax.Array,
    action_lo: jax.Array,
    action_hi: jax.Array,
) -> Sequence[jax.Array]:
    """
    Evaluates whether the policy is greedy w.r.t. the critic in terms of a distance metric.

    For some state in the policy buffer, we sample n_samples uniformly at random.
    The greedification for this state is the l2 distance between the action that
    maximizes the Q value and the action that maximizes the policy.

    Returns the metric for each state in the batch.
    """
    assert state.ndim == action_lo.ndim == action_hi.ndim == 1

    uniform_actions = jnp.asarray(
        np.random.uniform(
            low=action_lo,
            high=action_hi,
            size=(cfg.n_samples, agent.action_dim),
        ),
    )

    q_values = agent.get_active_values(state, uniform_actions)

    q = q_values.reduced_value
    chex.assert_shape(q, (cfg.n_samples, 1))

    max_actions_critic = get_max_action(uniform_actions, q)
    chex.assert_shape(max_actions_critic, (1, agent.action_dim))

    # Get log probabilities for the sampled actions from the actor.
    probs = agent.prob(state, uniform_actions)
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
        state: jax.Array,
        action_lo: jax.Array,
        action_hi: jax.Array,
    ):
    return agent_eval(
        app_state,
        agent,
        cfg_lens=lambda app_state: app_state.cfg.eval_cfgs.greed_dist_online,
        eval_fn=_greed_dist,
        metric_names=['greed_dist_online'],
        state=state,
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

@fail_gracefully()
def online_q_values_and_act_prob(
    app_state: AppState,
    agent: GreedyAC,
    state: jax.Array,
):
    """
    Logs the probability density function of the policy and the Q values.
    The entries are of the form (metric, x, y) where
    * metric tells us the action being varied (think x-axis)
    * x is a direct action
    * y is the probability or Q value, averaged over samples where the action in metric is set to x
        and the other actions are sampled from the policy. cfg.other_action_samples controls the number of samples
    """
    cfg = app_state.cfg.eval_cfgs.q_pdf_plots
    if not cfg.enabled:
        return

    x_axis_actions, probs, qs = q_values_and_act_prob(app_state, agent, state)
    chex.assert_rank(x_axis_actions, 1)
    chex.assert_shape([probs, qs], (agent.action_dim, len(x_axis_actions)))

    for a_dim_idx in range(agent.action_dim):
        # Write probability densities to evals table
        probs_measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(x_axis_actions, probs[a_dim_idx], strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"pdf_plot_action_{a_dim_idx}",
            value=probs_measure.model_dump_json(),
        )

        # Write q values to evals table
        qs_measure = XYEval(data=[
            XY(x=float(x), y=float(y))
            for x, y in zip(x_axis_actions, qs[a_dim_idx], strict=True)
        ])
        app_state.evals.write(
            agent_step=app_state.agent_step,
            evaluator=f"qs_plot_action_{a_dim_idx}",
            value=qs_measure.model_dump_json(),
        )

def q_values_and_act_prob(
    app_state: AppState,
    agent: GreedyAC,
    state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    cfg = app_state.cfg.eval_cfgs.q_pdf_plots

    chex.assert_rank(state, 1)

    rng = jax.random.PRNGKey(app_state.agent_step)
    dist = agent.get_dist(state)

    on_policy_actions: jax.Array = dist.sample(
        seed=rng,
        sample_shape=(cfg.primary_action_samples, cfg.other_action_samples),
    )
    chex.assert_shape(on_policy_actions, (cfg.primary_action_samples, cfg.other_action_samples, agent.action_dim))

    x_axis_actions = jnp.linspace(0, 1, cfg.primary_action_samples)

    qs: list[jax.Array] = []
    probs: list[jax.Array] = []
    for a_dim_idx in range(agent.action_dim):
        # Get actor probability densities over the current action dimension
        query_actions = on_policy_actions.at[:, :, a_dim_idx].set(jnp.expand_dims(x_axis_actions, 1))
        a_dim_probs = dist.prob(query_actions)
        chex.assert_shape(a_dim_probs, (cfg.primary_action_samples, cfg.other_action_samples))
        mean_a_dim_probs = a_dim_probs.mean(axis=1)
        probs.append(mean_a_dim_probs)

        # Get critic q values as the current action dimension is varied
        chex.assert_shape(query_actions, (cfg.primary_action_samples, cfg.other_action_samples, agent.action_dim))
        other_action_get_vals = jax_u.vmap(agent.get_active_values, in_axes=(None, 1), out_axes=-2)
        out = other_action_get_vals(state, query_actions)
        chex.assert_shape(out.reduced_value, (cfg.primary_action_samples, cfg.other_action_samples, 1))
        # remove the trailing value dim and avg over other_action dim
        a_dim_qs = out.reduced_value.squeeze(-1).mean(axis=-1)
        qs.append(a_dim_qs)

    return x_axis_actions, jnp.asarray(probs), jnp.asarray(qs)


class XY(BaseModel):
    x: float
    y: float

class XYEval(BaseModel):
    data: list[XY]
