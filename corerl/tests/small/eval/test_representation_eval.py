import jax.numpy as jnp

from corerl.eval.representation import RepresentationEval


def test_complexity_reduction_first_call():
    """
    First call should return 0.0 since Lrep becomes Lmax.

    Uses 3 states with known positions and representations to verify that the initial
    call sets Lmax and returns 0.0 (since 1 - Lrep/Lmax = 1 - 1 = 0).
    """
    rep_eval = RepresentationEval()

    states = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    reps = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])

    result = rep_eval.get_complexity_reduction(states, reps)

    assert result == 0.0
    assert rep_eval._lmax is not None
    assert rep_eval._lmax > 0


def test_complexity_reduction_subsequent_call():
    """
    Second call with lower Lrep should return 1 - Lrep/Lmax.

    First call sets Lmax, then second call with states that have lower complexity
    should return a positive value indicating complexity reduction.
    """
    rep_eval = RepresentationEval()

    states = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    reps_high = jnp.array([
        [0.0, 0.0],
        [2.0, 2.0],
        [4.0, 4.0],
    ])

    first_result = rep_eval.get_complexity_reduction(states, reps_high)
    assert first_result == 0.0

    reps_low = jnp.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    second_result = rep_eval.get_complexity_reduction(states, reps_low)

    assert 0.0 < second_result < 1.0
    assert second_result > first_result


def test_complexity_reduction_zero_variance():
    """
    All states identical should handle division by zero gracefully.

    When all states and representations are identical, the metric should handle
    the zero variance case without errors.
    """
    rep_eval_local = RepresentationEval()

    states = jnp.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ])

    reps = jnp.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    result = rep_eval_local.get_complexity_reduction(states, reps)

    assert isinstance(result, float)
    assert result >= 0.0


def test_dynamics_awareness_perfect():
    """
    Successor states identical to current states should give score near 1.0.

    When next_states == states (no dynamics), the successor distance is 0,
    so the score should be (random_dist - 0) / random_dist = 1.0.
    """
    rep_eval_local = RepresentationEval()

    states = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])

    next_states = states

    result = rep_eval_local.get_dynamics_awareness(states, next_states)

    assert result >= 0.9


def test_dynamics_awareness_none():
    """
    Random permutation of states as successors should give low score.

    When successor distances are comparable to random distances, score should be low.
    In this case the successors are actually further than average, giving negative score.
    """
    rep_eval_local = RepresentationEval()

    states = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    next_states = jnp.array([
        [2.0, 2.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])

    result = rep_eval_local.get_dynamics_awareness(states, next_states)

    assert -1.0 < result < 0.2


def test_dynamics_awareness_intermediate():
    """
    Partial dynamics correlation should give score between 0 and 1.

    When successors are closer than random but not identical, score should be positive but less than 1.
    """
    rep_eval_local = RepresentationEval()

    states = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])

    next_states = jnp.array([
        [0.1, 0.0],
        [1.1, 0.0],
        [2.1, 0.0],
        [3.1, 0.0],
    ])

    result = rep_eval_local.get_dynamics_awareness(states, next_states)

    assert 0.0 < result < 1.0


def test_diversity_high():
    """
    Similar values with distant representations should give moderate-to-high diversity score.

    When states have similar values but very different representations,
    diversity should be positive. The metric uses a 1e-2 denominator offset
    which affects the score.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])

    values = jnp.array([1.0, 1.1, 1.05, 1.02])

    result = rep_eval_local.get_diversity(reps, values)

    assert result > 0.5


def test_diversity_low():
    """
    Close representations for close values should give low diversity score.

    When representations are specialized to values (close reps for close values),
    diversity should be low.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [1.1, 1.1],
    ])

    values = jnp.array([0.0, 0.1, 1.0, 1.1])

    result = rep_eval_local.get_diversity(reps, values)

    assert result < 0.5


def test_diversity_range():
    """
    Verify diversity score is always in [0, 1] range.

    Tests edge cases to ensure the diversity metric is properly bounded.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    values = jnp.array([5.0, 10.0, 15.0])

    result = rep_eval_local.get_diversity(reps, values)

    assert 0.0 <= result <= 1.0


def test_orthogonality_perfect():
    """
    Orthogonal vectors should give score = 1.0.

    When feature vectors are perfectly orthogonal (like identity matrix rows),
    the orthogonality score should be 1.0.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    result = rep_eval_local.get_orthogonality(reps)

    assert result > 0.99


def test_orthogonality_parallel():
    """
    All vectors in same direction should give score near 0.0.

    When all feature vectors point in the same direction (parallel),
    orthogonality should be 0.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
    ])

    result = rep_eval_local.get_orthogonality(reps)

    assert result < 0.01


def test_orthogonality_intermediate():
    """
    Mixed orthogonal and parallel components should give intermediate score.

    When vectors have some orthogonal and some parallel components,
    the score should be between 0 and 1.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [1.0, 0.0],
        [0.5, 0.866],
        [0.0, 1.0],
    ])

    result = rep_eval_local.get_orthogonality(reps)

    assert 0.3 < result < 0.9


def test_sparsity_fully_sparse():
    """
    All zeros should give score = 1.0.

    When all feature values are zero (fully sparse), sparsity should be 1.0.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    result = rep_eval_local.get_sparsity(reps)

    assert result == 1.0


def test_sparsity_fully_dense():
    """
    No zeros should give score = 0.0.

    When no feature values are zero (fully dense), sparsity should be 0.0.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    result = rep_eval_local.get_sparsity(reps)

    assert result == 0.0


def test_sparsity_half_sparse():
    """
    50% zeros should give score ≈ 0.5.

    When half of the features are zero, sparsity should be approximately 0.5.
    """
    rep_eval_local = RepresentationEval()

    reps = jnp.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ])

    result = rep_eval_local.get_sparsity(reps)

    assert 0.4 < result < 0.6
