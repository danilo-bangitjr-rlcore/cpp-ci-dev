import jax.numpy as jnp
import numpy as np
from ml_instrumentation.Collector import Collector

from agent.gac import GreedyAC


def ac_eval(collector: Collector, agent: GreedyAC, state: np.ndarray):
    x_axis_actions = 101
    on_policy_samples = 5

    # Actor and Critic evaluated at evenly spaced points along the x-axis
    linspaced_actions = jnp.linspace(start=0, stop=1, num=x_axis_actions, endpoint=True)
    repeat_linspace = linspaced_actions.repeat(on_policy_samples)

    # To evaluate critic at a given point along x-axis, use average over sampled actions for remaining action dims
    repeat_state = jnp.tile(state, (on_policy_samples, 1))
    on_policy_actions = agent.get_actions(repeat_state)
    repeat_on_policy = jnp.tile(on_policy_actions, (x_axis_actions, 1))

    # Actor
    actor_probs = agent.get_probs(agent.agent_state.actor.params, state, linspaced_actions)

    # Proposal
    proposal_probs = agent.get_probs(agent.agent_state.proposal.params, state, linspaced_actions)

    for a_dim in range(agent.action_dim):
        constructed_actions = repeat_on_policy.at[:,a_dim].set(repeat_linspace)

        # Critic
        q_vals = agent.get_action_values(agent.agent_state.critic.params, state, constructed_actions)
        reshaped_q_vals = q_vals.reshape((x_axis_actions, on_policy_samples))
        state_vals = reshaped_q_vals.mean(axis=1)

        # Actor
        a_dim_actor_probs = actor_probs[:,a_dim]

        # Proposal
        a_dim_proposal_probs = proposal_probs[:,a_dim]

        # Write to db
        for i in range(x_axis_actions):
            collector.collect(f"ac_actions_{a_dim}_{i}", float(state_vals[i]))
            collector.collect(f"actor_pdf_plot_action_{a_dim}_{i}", float(a_dim_actor_probs[i]))
            collector.collect(f"proposal_pdf_plot_action_{a_dim}_{i}", float(a_dim_proposal_probs[i]))
            collector.collect(f"qs_plot_action_{a_dim}_{i}", float(state_vals[i]))
