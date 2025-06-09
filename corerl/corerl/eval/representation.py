
import jax
import jax.numpy as jnp

from corerl.agent.greedy_ac import GreedyAC
from corerl.agent.utils import get_sampled_qs
from corerl.state import AppState


class RepresentationEval:
    """
    Various metrics on learned representations.
    Currently implements:
    - Complexity Reduction: Measures how much the representation helps reduce
      the required complexity in the value function
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self._lmax: float | None = None

    def get_complexity_reduction(
        self,
        states: jax.Array,
        values: jax.Array,
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
        state_diffs = jnp.sqrt(jnp.sum((states[:, None] - states[None, :])**2, axis=-1))

        # calculate pairwise value differences
        value_diffs = jnp.abs(values[:, None] - values[None, :])

        # get upper triangular indices (excluding the diagonal)
        mask = jnp.triu(jnp.ones((n, n)), k=1).astype(bool)

        # get ratios for all pairs
        ratios = value_diffs[mask] / (state_diffs[mask] + 1e-8)

        # calculate Lrep as average of ratios
        lrep = 2 * jnp.mean(ratios) / (n * (n-1))

        if self._lmax is None or lrep > self._lmax:
            self._lmax = lrep

        return 1 - (lrep / self._lmax)

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

        batch = batches[0]
        states = batch.prior.state
        action_lo = batch.prior.action_lo
        action_hi = batch.prior.action_hi

        qr = get_sampled_qs(
            states=states,
            action_lo=action_lo,
            action_hi=action_hi,
            n_samples=100,
            sampler=agent.get_uniform_actions,
            critic=agent,
        )

        jax_states = jnp.array(states.cpu().numpy())
        jax_q_values = jnp.array(qr.q_values.mean(dim=1).cpu().numpy())

        complexity_reduction = self.get_complexity_reduction(
            jax_states,
            jax_q_values,
        )

        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="representation_complexity_reduction",
            value=float(complexity_reduction),
        )



