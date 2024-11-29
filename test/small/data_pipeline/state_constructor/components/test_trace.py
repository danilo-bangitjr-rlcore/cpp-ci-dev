import numpy as np
import pandas as pd
from corerl.data_pipeline.state_constructors.interface import TransformCarry
from corerl.data_pipeline.state_constructors.components.trace import (
    compute_trace_with_nan, TraceConfig, TraceConstructor, TraceTemporalState
)

def test_compute_trace1():
    """
    Given a sequence with a temporal break, get back
    a trace that restarts after the trace.
    """
    data = np.array([1, 2, 3, 4, np.nan, 1, 2, 3, 4])

    # expect a (n_samples, n_traces) == (9, 1) array
    expected = np.array([[
        # first sequence until nan
        1., 1.9, 2.89, 3.889,
        # nan break
        np.nan,
        # second sequence starts over again
        1., 1.9, 2.89, 3.889,
    ]]).T

    trace, mu = compute_trace_with_nan(
        data,
        decays=np.array([0.1]),
    )

    assert trace.shape == (9, 1)
    assert np.allclose(expected, trace, equal_nan=True)

    assert mu is not None
    assert mu.shape == (1,)
    assert np.isclose(mu, [3.889])


def test_compute_trace2():
    """
    Given a sequence that starts with some np.nan,
    create a trace starting from first non-nan idx
    """
    data = np.array([np.nan, np.nan, 1, 2, 3, 4])

    # expect a (n_samples, n_traces) == (6, 1) array
    expected = np.array([[
        np.nan, np.nan,
        1., 1.9, 2.89, 3.889,
    ]]).T

    trace, mu = compute_trace_with_nan(
        data,
        decays=np.array([0.1]),
    )

    assert trace.shape == (6, 1)
    assert np.allclose(expected, trace, equal_nan=True)

    assert mu is not None
    assert mu.shape == (1,)
    assert np.isclose(mu, [3.889])


def test_compute_trace3():
    """
    Given a sequence that ends with some np.nan,
    create a trace and return a None temporal state (mu)
    """
    data = np.array([1, 2, 3, 4, np.nan, np.nan])

    # expect a (n_samples, n_traces) == (6, 1) array
    expected = np.array([[
        1., 1.9, 2.89, 3.889,
        np.nan, np.nan,
    ]]).T

    trace, mu = compute_trace_with_nan(
        data,
        decays=np.array([0.1]),
    )

    assert trace.shape == (6, 1)
    assert np.allclose(expected, trace, equal_nan=True)

    assert mu is None


def test_compute_multiple_traces():
    """
    Specifying multiple trace decays produces multiple outputs
    from a single input stream.
    """
    data = np.array([np.nan, 1, 2, 3, np.nan, 1, 2, np.nan])

    # expect a (n_samples, n_traces) == (8, 2) array
    expected = np.array([
        # decay=0.1
        [np.nan, 1., 1.9, 2.89, np.nan, 1., 1.9, np.nan],
        # decay=0.01
        [np.nan, 1., 1.99, 2.9899, np.nan, 1., 1.99, np.nan],
    ]).T

    trace, mu = compute_trace_with_nan(
        data,
        decays=np.array([0.1, 0.01]),
    )

    assert trace.shape == (8, 2)
    assert np.allclose(expected, trace, equal_nan=True)

    assert mu is None


def test_trace_first_data():
    """
    First invocation of the TraceConstructor
    (or more generally invocation with no prior temporal state)
    yields a modified agent_state dataframe with traces added to
    each column and a new temporal state.
    """
    raw_obs = pd.DataFrame({
        'obs_1': np.array([np.nan, 1, 2, 3, np.nan, 1, 2]),
        'obs_2': np.array([1, 2, np.nan, 1, 2, 3, np.nan]),
    })

    carry = TransformCarry(
        obs=raw_obs,
        agent_state=raw_obs,
        tag='obs',
    )

    trace_sc = TraceConstructor(
        cfg=TraceConfig(
            trace_values=[0.1, 0.01],
        ),
    )

    new_carry, new_ts = trace_sc(carry, ts=None)

    assert set(new_carry.agent_state.columns) == {
        'obs_1_trace-0.1',
        'obs_1_trace-0.01',
        'obs_2_trace-0.1',
        'obs_2_trace-0.01',
    }
    assert new_carry.agent_state.shape == (7, 4)
    assert np.allclose(
        new_carry.agent_state['obs_1_trace-0.1'],
        np.array([np.nan, 1., 1.9, 2.89, np.nan, 1., 1.9]),
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.agent_state['obs_1_trace-0.01'],
        np.array([np.nan, 1., 1.99, 2.9899, np.nan, 1., 1.99]),
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.agent_state['obs_2_trace-0.1'],
        np.array([1., 1.9, np.nan, 1., 1.9, 2.89, np.nan]),
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.agent_state['obs_2_trace-0.01'],
        np.array([1., 1.99, np.nan, 1., 1.99, 2.9899, np.nan]),
        equal_nan=True,
    )
    assert new_ts.mu is not None

    # obs_1 did not end in np.nan, so has a carry state
    mu_obs1 = new_ts.mu['obs_1']
    assert mu_obs1 is not None
    assert np.allclose(
        mu_obs1,
        np.array([1.9, 1.99]),
    )

    # obs_2 did end in nan, so does not have a carry state
    mu_obs2 = new_ts.mu['obs_2']
    assert mu_obs2 is None


def test_trace_temporal_state():
    """
    Given a sequence of data _and_ an initial starting state
    in the temporal state, the trace continues from the
    temporal state.
    """
    raw_obs = pd.DataFrame({
        'obs_1': np.array([np.nan, 1, 2, 3, np.nan, 1, 2]),
        'obs_2': np.array([1, 2, np.nan, np.nan, np.nan, np.nan, np.nan]),
    })

    carry = TransformCarry(
        obs=raw_obs,
        agent_state=raw_obs,
        tag='obs',
    )

    trace_sc = TraceConstructor(
        cfg=TraceConfig(
            trace_values=[0.1, 0.01],
        ),
    )

    ts = TraceTemporalState(
        mu={
            'obs_1': None,
            'obs_2': np.array([20., 40.]),
        },
    )

    new_carry, new_ts = trace_sc(carry, ts)

    assert np.allclose(
        new_carry.agent_state['obs_1_trace-0.1'],
        [np.nan, 1., 1.9, 2.89, np.nan, 1., 1.9],
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.agent_state['obs_1_trace-0.01'],
        [np.nan, 1., 1.99, 2.9899, np.nan, 1., 1.99],
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.agent_state['obs_2_trace-0.1'],
        [2.9, 2.09, np.nan, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.agent_state['obs_2_trace-0.01'],
        [1.39, 1.9939, np.nan, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
