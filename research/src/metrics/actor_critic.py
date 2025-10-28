from functools import partial

import chex
import jax.numpy as jnp
import lib_utils.jax as jax_u
from lib_agent.buffer.datatypes import State

from agent.gac import GreedyAC


def ac_eval(agent: GreedyAC, state: State):
    num_x_axis_actions = 101
    on_policy_samples = 5

    # Actor and Critic evaluated at evenly spaced points along the x-axis
    x_axis_actions_ = jnp.linspace(start=0, stop=1, num=num_x_axis_actions, endpoint=True)
    x_axis_actions = jnp.expand_dims(x_axis_actions_, axis=1)
    x_axis_actions = jnp.repeat(x_axis_actions, repeats=on_policy_samples, axis=1)
    chex.assert_shape(x_axis_actions, (num_x_axis_actions, on_policy_samples))

    # To evaluate critic at a given point along x-axis, use average over sampled actions for remaining action dim
    on_policy_actions = agent.get_actions(state, n_samples=on_policy_samples)
    action_dim = state.a_hi.shape[-1]
    chex.assert_shape(on_policy_actions, (on_policy_samples, action_dim))
    on_policy_actions = jnp.expand_dims(on_policy_actions, axis=0)
    chex.assert_shape(on_policy_actions, (1, on_policy_samples, action_dim))
    on_policy_actions = jnp.repeat(on_policy_actions, repeats=num_x_axis_actions, axis=0)
    chex.assert_shape(on_policy_actions, (num_x_axis_actions, on_policy_samples, action_dim))

    return_q_vals = []
    return_a_probs = []

    for a_dim in range(agent.action_dim):
        constructed_actions = on_policy_actions.at[:, :, a_dim].set(x_axis_actions)

        # q vals
        q_f = partial(agent.get_action_values, state)
        q_vals = jax_u.multi_vmap(q_f, levels=2)(constructed_actions).squeeze()
        chex.assert_shape(q_vals, (num_x_axis_actions, on_policy_samples))

        # actor
        ap_f = partial(agent.get_probs, agent.agent_state.actor.actor.params, state)
        actor_probs = jax_u.multi_vmap(ap_f, levels=1)(constructed_actions).squeeze()
        chex.assert_shape(actor_probs, (num_x_axis_actions, on_policy_samples))

        return_q_vals.append(jnp.expand_dims(q_vals.mean(axis=1), 1))
        return_a_probs.append(jnp.expand_dims(actor_probs.mean(axis=1), 1))

    x_axis_actions = jnp.expand_dims(x_axis_actions_, axis=1)
    x_axis_actions = jnp.repeat(x_axis_actions, repeats=action_dim, axis=1)

    return jnp.concat(return_q_vals, axis=1), jnp.concat(return_a_probs, axis=1), x_axis_actions
