import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u

from corerl.agent.greedy_ac import GreedyAC
from corerl.state import AppState


class RepresentationEval:
    """
    Various metrics on learned representations.
    Currently implements:
    - Complexity Reduction: Measures how much the representation helps reduce
      the required complexity in the value function
    - Dynamics Awareness: Measures how well the representation captures state dynamics
      by comparing distances between states and their successors vs random states
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self._lmax: float | None = None

        self._rng = jax.random.PRNGKey(0)

    def get_complexity_reduction(
        self,
        states: jax.Array,
        reps: jax.Array,
    ) -> float:
        """
        Calculate complexity reduction metric for the current batch.

        This metric reflects how much the representation facilitates simplicity of the learned
        value function on top of those features. If complexity is small, the features encode
        much of the non-linearity needed. The metric is based on the Lipschitz constant L
        where dv,i,j/ds,i,j ≤ L for any pair of states (i,j).  Higher values indicate better
        complexity reduction, meaning the representation better handles the non-linearity needed
        for the value function. This is beneficial because:
        1. We have a longer initial learning phase to obtain the representation
        2. The value function on top of the representation should be fast to learn
        3. It facilitates better value transfer and model learning

        The metric is computed by:
        1. Measuring dv,i,j/ds,i,j for all pairs (i,j) in the batch
        2. Computing Lrep as the average of these ratios: Lrep = 2/(N(N-1)) * Σ(dv,i,j/ds,i,j)
        3. Normalizing by Lmax (maximum Lrep seen so far)
        4. Computing Complexity Reduction = 1 - Lrep/Lmax
        """
        n = states.shape[0]

        # calculate pairwise state differences
        state_diffs = jnp.sqrt(jnp.sum((states[:, None, :] - states[None, :, :])**2, axis=-1))

        # calculate pairwise representation differences
        rep_diffs = jnp.sqrt(jnp.sum((reps[:, None, :] - reps[None, :, :])**2, axis=-1))

        # get upper triangular indices (excluding the diagonal)
        mask = jnp.triu(jnp.ones((n, n)), k=1).astype(bool)

        # get ratios for all pairs
        ratios = rep_diffs[mask] / (state_diffs[mask] + 1e-8)

        # calculate Lrep as average of ratios
        lrep = float(2 * jnp.mean(ratios) / (n * (n-1)))

        if self._lmax is None or lrep > self._lmax:
            self._lmax = lrep

        if self._lmax == 0:
            return 0.0

        return 1 - (lrep / self._lmax)

    def get_dynamics_awareness(
        self,
        states: jax.Array,
        next_states: jax.Array,
    ) -> float:
        """
        Calculate dynamics awareness metric for the current batch.

        This metric measures how well the representation captures state dynamics by comparing
        distances between states and their successors vs random states. A higher score indicates
        that successor states are closer in representation space than random states, suggesting
        the representation is dynamics-aware.

        The metric is computed by:
        1. For each state i:
           - Calculate distance to its successor state φ'ᵢ
           - Calculate distance to a random state φⱼ
        2. Sum these distances across all states
        3. Compute: (∑ᵢ ||φᵢ - φⱼ|| - ∑ᵢ ||φᵢ - φ'ᵢ||) / (∑ᵢ ||φᵢ - φⱼ||)

        Returns a score between -1 and 1, where:
        - 1 means perfect dynamics awareness (successors are much closer than random states)
        - 0 means no dynamics awareness (successors are as far as random states)
        - -1 means anti-dynamics awareness (successors are much further than random states)
        """
        n = states.shape[0]

        # sample random state indices for each state
        rng = jax.random.PRNGKey(0)
        random_indices = jax.random.permutation(rng, n)

        # calculate distances to random states
        random_dists = jnp.sqrt(jnp.sum((states - states[random_indices])**2, axis=-1))
        total_random_dist = jnp.sum(random_dists)

        # calculate distances to successor states
        successor_dists = jnp.sqrt(jnp.sum((states - next_states)**2, axis=-1))
        total_successor_dist = jnp.sum(successor_dists)

        # compute dynamics awareness score
        dynamics_awareness = (total_random_dist - total_successor_dist) / (total_random_dist + 1e-8)

        return float(dynamics_awareness)

    def get_diversity(
        self,
        reps: jax.Array,
        values: jax.Array,
    ) -> float:
        """
        Calculate diversity metric for the current batch.

        This metric measures the diversity of the learned representation, which is the opposite of specialization.
        It compares the normalized pairwise distances in representation to the normalized pairwise distances in value.
        If two states have similar values but are mapped to distant representations, the diversity is high.
        If the representation is specialized to the value function, diversity is low.

        The metric is computed by:
        1. For all pairs of states (i, j):
           - Compute ds_{i,j}: L2 distance between representations
           - Compute dv_{i,j}: absolute difference between values
        2. Normalize ds_{i,j} and dv_{i,j} by their respective maxima
        3. For each pair, compute the ratio:
             ratio_{i,j} = (dv_{i,j} / max dv) / (ds_{i,j} / max ds + 1e-2)
           and cap at 1
        4. Average the ratio over all pairs and subtract from 1:
             Diversity = 1 - mean(ratio_{i,j})

        Returns a score between 0 and 1, where higher is more diverse.
        """
        # calculate distance in representation and values
        rep_diffs = jnp.sqrt(jnp.sum((reps[:, None] - reps[None, :])**2, axis=-1))
        value_diffs = jnp.abs(values[:, None] - values[None, :])
        max_rep_diff = jnp.max(rep_diffs)
        max_value_diff = jnp.max(value_diffs)

        rep_diffs_norm = rep_diffs / (max_rep_diff + 1e-8)
        value_diffs_norm = value_diffs / (max_value_diff + 1e-8)

        ratio = value_diffs_norm / (rep_diffs_norm + 1e-2)
        ratio = jnp.minimum(ratio, 1.0)
        diversity = 1.0 - jnp.mean(ratio)
        return float(diversity)

    def get_orthogonality(
        self,
        reps: jax.Array,
    ) -> float:
        """
        Calculate orthogonality metric for the current batch.

        This metric measures how orthogonal the feature vectors are to each other, normalized by their magnitudes.
        Higher orthogonality means more feature vectors are orthogonal to each other, indicating:
        1. Less redundancy in the representation
        2. More distributed features
        3. Minimal interference between features

        The metric is computed by:
        1. For all pairs of states (i, j) where i < j:
           - Compute dot product between feature vectors: <φi, φj>
           - Normalize by magnitudes: |<φi, φj>| / (||φi||2 * ||φj||2)
        2. Average over all pairs and subtract from 1:
           Orthogonality = 1 - 2/(N(N-1)) * Σ |<φi, φj>| / (||φi||2 * ||φj||2)

        Returns a score between 0 and 1, where 1 is perfect orthogonality
        """
        n = reps.shape[0]

        magnitudes = jnp.sqrt(jnp.sum(reps**2, axis=-1))

        # compute dot products between all pairs
        dot_products = jnp.matmul(reps, reps.T)
        mask = jnp.triu(jnp.ones((n, n)), k=1).astype(bool)
        mag_products = magnitudes[:, None] * magnitudes[None, :]

        # normalize dot products by magnitudes
        normalized_dots = jnp.abs(dot_products[mask]) / (mag_products[mask] + 1e-8)

        # compute orthogonality score
        orthogonality = 1.0 - jnp.mean(normalized_dots)

        return float(orthogonality)

    def get_sparsity(
        self,
        reps: jax.Array,
    ) -> float:
        """
        Calculate sparsity metric for the current batch.

        This metric measures how many features are inactive across all states.
        Higher sparsity means fewer features are active for each state, which means
        1. More efficient querying and updating
        2. Better feature specialization
        3. Reduced interference between features

        The metric is computed by:
        1. For each state i and feature j:
           - Check if feature is inactive: 1(φi,j = 0) within tolerance 1e-10
        2. Average over all states and features:
           Sparsity = 1/(d*N) * Σᵢ Σⱼ 1(φi,j = 0)

        Returns a score between 0 and 1, where 1 means all features are inactive
        """
        inactive_features = jnp.abs(reps) < 1e-10
        sparsity = jnp.mean(inactive_features)

        return float(sparsity)

    def evaluate(
        self,
        app_state: AppState,
        agent: GreedyAC,
    ) -> None:
        """
        Evaluate representation metrics using states from the policy buffer.
        """
        if not agent.is_policy_buffer_sampleable:
            return

        batches = agent.sample_policy_buffer()
        if not batches:
            return

        batch = jax.tree.map(lambda x: x[0], batches)
        states = batch.state
        next_states = batch.next_state
        action_lo = batch.action_lo
        action_hi = batch.action_hi

        n_samples = 100
        num_states = states.shape[0]
        action_dim = action_lo.shape[1]
        self._rng, sample_rng = jax.random.split(self._rng)
        q_rngs = jax.random.split(sample_rng, (num_states, n_samples))

        # sample uniform actions for each state
        sampled_actions = jax.random.uniform(
            sample_rng,
            shape=(num_states, n_samples, action_dim),
            minval=action_lo[:, None, :],
            maxval=action_hi[:, None, :],
        )

        get_ens_state_reps = jax_u.vmap_only(agent.critic.get_representations, ["params"])
        ens_state_reps = get_ens_state_reps(
            agent._critic_state.params,
            q_rngs,
            states,
            sampled_actions,
        )
        chex.assert_shape(ens_state_reps, (agent.ensemble, agent._actor_buffer.batch_size, n_samples, None))
        ens_next_state_reps = get_ens_state_reps(
            agent._critic_state.params,
            q_rngs,
            next_states,
            sampled_actions,
        )
        chex.assert_shape(ens_next_state_reps, (agent.ensemble, agent._actor_buffer.batch_size, n_samples, None))

        # average representations for each state: avg over ensemble and action samples
        mean_state_reps = ens_state_reps.mean(axis=(0, 2))
        mean_next_state_reps = ens_next_state_reps.mean(axis=(0, 2))

        # get values for each state (using Q-values for the current policy)
        qs = jax_u.vmap_only(agent.critic.get_values, ["params"])(
            agent._critic_state.params,
            q_rngs,
            states,
            sampled_actions,
        ).q
        chex.assert_shape(qs, (agent.ensemble, agent._actor_buffer.batch_size, n_samples, 1))
        mean_qs = qs.mean(axis=(0, 2)).squeeze(-1) # avg over ensemble members and action samples

        complexity_reduction = self.get_complexity_reduction(
            states,
            mean_state_reps,
        )

        # calculate dynamics awareness using representations
        dynamics_awareness = self.get_dynamics_awareness(
            mean_state_reps,
            mean_next_state_reps,
        )
        diversity = self.get_diversity(
            mean_state_reps,
            mean_qs,
        )
        orthogonality = self.get_orthogonality(
            mean_state_reps,
        )
        sparsity = self.get_sparsity(
            mean_state_reps,
        )
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="representation_complexity_reduction",
            value=float(complexity_reduction),
        )
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="representation_dynamics_awareness",
            value=float(dynamics_awareness),
        )
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="representation_diversity",
            value=float(diversity),
        )
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="representation_orthogonality",
            value=float(orthogonality),
        )
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="representation_sparsity",
            value=float(sparsity),
        )
