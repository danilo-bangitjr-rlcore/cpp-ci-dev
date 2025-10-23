import jax
import jax.numpy as jnp
from chex import ArrayTree
from lib_utils.named_array import NamedArray

from corerl.agent.greedy_ac import GreedyAC


def test_critic_update_reduces_loss_over_iterations(populated_agent: GreedyAC):
    """
    Verify critic learning: repeated updates should reduce TD error.

    Tests that the critic network learns to better predict value functions
    by checking that loss decreases over multiple training iterations.
    """
    initial_losses = populated_agent.update_critic()
    initial_avg_loss = sum(initial_losses) / len(initial_losses)

    for _ in range(10):
        populated_agent.update_critic()

    final_losses = populated_agent.update_critic()
    final_avg_loss = sum(final_losses) / len(final_losses)

    assert final_avg_loss < initial_avg_loss, (
        f"Critic should learn: final loss {final_avg_loss:.4f} "
        f"should be less than initial loss {initial_avg_loss:.4f}"
    )


def test_actor_update_improves_policy_objective(populated_agent: GreedyAC):
    """
    Verify actor learning: policy updates should decrease actor loss.

    Tests that the actor network learns to select actions with higher
    Q-values by checking that the actor loss decreases over iterations.
    """
    initial_loss = populated_agent.update_actor()

    for _ in range(10):
        populated_agent.update_actor()

    final_loss = populated_agent.update_actor()

    assert final_loss < initial_loss, (
        f"Actor should learn: final loss {final_loss:.4f} "
        f"should be less than initial loss {initial_loss:.4f}"
    )


def test_ensemble_aggregation_reduces_variance(greedy_ac_agent: GreedyAC):
    """
    Verify ensemble value aggregation reduces uncertainty.

    Tests that aggregating values from multiple ensemble members
    produces estimates with lower variance than individual members,
    validating the ensemble's uncertainty reduction property.
    """
    state = NamedArray.unnamed(jnp.zeros(greedy_ac_agent.state_dim))
    action = jnp.zeros(greedy_ac_agent.action_dim)

    result = greedy_ac_agent.get_active_values(state, action)

    assert result.ensemble_values.shape[0] == 2, "Should have 2 ensemble members"

    individual_variances = jnp.var(result.ensemble_values)
    ensemble_variance = result.ensemble_variance

    assert jnp.all(ensemble_variance <= individual_variances + 1e-6), (
        "Ensemble aggregation should not increase variance"
    )


def test_get_active_values_returns_ensemble_statistics(greedy_ac_agent: GreedyAC):
    """
    Verify get_active_values computes correct ensemble statistics.

    Tests that value estimates include both the aggregated mean value
    and the ensemble variance, which are used for uncertainty-aware
    decision making.
    """
    state = NamedArray.unnamed(jnp.ones(greedy_ac_agent.state_dim))
    action = jnp.ones(greedy_ac_agent.action_dim)

    result = greedy_ac_agent.get_active_values(state, action)

    assert result.ensemble_values.shape == (2, 1), "Should return values from 2 critics"
    assert result.reduced_value.shape == (1,), "Reduced value should always be 1 (`q(s, a)`)"
    assert result.ensemble_variance.shape == (1,), "Variance should always be 1 (`q(s, a)`)"


def test_training_updates_critic_and_actor_parameters(populated_agent: GreedyAC):
    """
    Verify training updates both critic and actor network parameters.

    Tests that calling update_critic and update_actor modifies the
    network parameters, confirming that gradient descent is applied.
    """
    initial_critic_params = populated_agent._critic_state.params
    initial_actor_params = populated_agent._actor_state.actor.params

    populated_agent.update_critic()
    populated_agent.update_actor()

    updated_critic_params = populated_agent._critic_state.params
    updated_actor_params = populated_agent._actor_state.actor.params

    def params_changed(p1: ArrayTree, p2: ArrayTree):
        leaves1 = jnp.concatenate([x.flatten() for x in jax.tree_util.tree_leaves(p1)])
        leaves2 = jnp.concatenate([x.flatten() for x in jax.tree_util.tree_leaves(p2)])
        return not jnp.allclose(leaves1, leaves2)

    assert params_changed(initial_critic_params, updated_critic_params), (
        "Critic parameters should change after update_critic()"
    )
    assert params_changed(initial_actor_params, updated_actor_params), (
        "Actor parameters should change after update_actor()"
    )
