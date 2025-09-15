import numpy as np
import pandas as pd

from corerl.data_pipeline.transforms.interface import TransformCarry
from corerl.data_pipeline.transforms.trace import (
    TraceConfig,
    TraceConstructor,
    TraceData,
    TraceParams,
    TraceTemporalState,
    compute_trace_with_nan,
)


def test_compute_trace1():
    """
    Given a sequence with a temporal break, get back
    a trace that restarts after the trace.
    """
    data = np.array([1, 2, 3, 4])

    # expect a (n_samples, n_traces) == (4, 1) array
    expected = np.array([[
        1., 1.9, 2.89, 3.889,
    ]]).T

    trace_params = TraceParams(decays=np.array([0.1]), missing_tol=1.0)
    trace_data = TraceData(trace=np.ones(1)*np.nan, obs=data, quality=np.zeros(1))
    trace, trace_data = compute_trace_with_nan(trace_params, trace_data)

    assert trace.shape == (4, 1)
    assert np.allclose(expected, trace, equal_nan=True)

    assert trace_data.trace.shape == (1,)
    assert np.isclose(trace_data.trace, [3.889])


def test_compute_multiple_traces():
    """
    Specifying multiple trace decays produces multiple outputs
    from a single input stream.
    """
    data = np.array([1, 2, 3])

    # expect a (n_samples, n_traces) == (8, 2) array
    expected = np.array([
        # decay=0.1
        [1., 1.9, 2.89],
        # decay=0.01
        [1., 1.99, 2.9899],
    ]).T

    trace_params = TraceParams(decays=np.array([0.1, 0.01]), missing_tol=1.0)
    trace_data = TraceData(trace=np.ones(2)*np.nan, obs=data, quality=np.zeros(2))
    trace, trace_data = compute_trace_with_nan(trace_params, trace_data)

    assert trace.shape == (3, 2)
    assert np.allclose(expected, trace, equal_nan=True)


def test_trace_first_data():
    """
    First invocation of the TraceConstructor
    (or more generally invocation with no prior temporal state)
    yields a modified agent_state dataframe with traces added to
    each column and a new temporal state.
    """
    raw_obs = pd.DataFrame({
        'obs_1': np.array([1, 2, 3]),
        'obs_2': np.array([1, 2, 3]),
    })

    carry = TransformCarry(
        obs=raw_obs,
        transform_data=raw_obs,
        tag='obs',
    )

    trace_sc = TraceConstructor(
        cfg=TraceConfig(
            trace_values=[0.1, 0.01],
            missing_tol=1.0,
        ),
    )

    new_carry, new_ts = trace_sc(carry, ts=None)

    assert set(new_carry.transform_data.columns) == {
        'obs_1_trace-0.1',
        'obs_1_trace-0.01',
        'obs_2_trace-0.1',
        'obs_2_trace-0.01',
    }
    assert new_carry.transform_data.shape == (3, 4)
    assert np.allclose(
        new_carry.transform_data['obs_1_trace-0.1'],
        np.array([1., 1.9, 2.89]),
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.transform_data['obs_1_trace-0.01'],
        np.array([1., 1.99, 2.9899]),
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.transform_data['obs_2_trace-0.1'],
        np.array([1., 1.9, 2.89]),
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.transform_data['obs_2_trace-0.01'],
        np.array([1., 1.99, 2.9899]),
        equal_nan=True,
    )
    assert new_ts.trace is not None


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
        transform_data=raw_obs,
        tag='obs',
    )

    trace_sc = TraceConstructor(
        cfg=TraceConfig(
            trace_values=[0.1, 0.01],
            missing_tol=1.0,
        ),
    )

    ts = TraceTemporalState(
        trace={
            'obs_1': np.ones(2) * np.nan,
            'obs_2': np.array([20., 40.]),
        },
    )

    new_carry, _ = trace_sc(carry, ts)

    assert np.allclose(
        new_carry.transform_data['obs_1_trace-0.1'],
        [np.nan, 1., 1.9, 2.89, 2.89, 1.189, 1.9189],
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.transform_data['obs_1_trace-0.01'],
        [np.nan, 1., 1.99, 2.9899, 2.9899, 1.019899, 1.990199],
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.transform_data['obs_2_trace-0.1'],
        [2.9, 2.09, 2.09, 2.09, 2.09, 2.09, 2.09],
        equal_nan=True,
    )
    assert np.allclose(
        new_carry.transform_data['obs_2_trace-0.01'],
        [1.39, 1.9939, 1.9939, 1.9939, 1.9939, 1.9939, 1.9939],
        equal_nan=True,
    )


def test_trace_warmup():
    """
    Traces remain NaN for some number of initial steps based on decay and missing_tol
    """
    data = np.array([1, 2, 3, np.nan, 1, 2, np.nan])

    # expect a (n_samples, n_traces) == (7, 3) array
    expected = np.array([
        [np.nan, 1.3, 1.81, np.nan, 1.567, 1.6969, 1.6969],
        [np.nan, np.nan, np.nan, np.nan, 1.448, 1.5584, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]).T

    trace_params = TraceParams(decays=np.array([0.7, 0.8, 0.99]), missing_tol=0.5)
    trace_data = TraceData(trace=np.ones(3)*np.nan, obs=data, quality=np.zeros(3))
    trace, trace_data = compute_trace_with_nan(trace_params, trace_data)

    assert trace.shape == (7, 3)
    assert np.allclose(expected, trace, equal_nan=True)

def test_trace_nantol():
    """
    Test that traces can tolerate some NaNs midstream with large decay and sufficient missing_tol
    """
    data = np.array([1, 2, np.nan, 3])

    expected = np.array([
        [1., 1.25, 1.25, 1.6875],
    ]).T

    trace_params = TraceParams(decays=np.array([0.75]), missing_tol=0.9)
    trace_data = TraceData(trace=np.ones(1)*np.nan, obs=data, quality=np.zeros(1))
    trace, trace_data = compute_trace_with_nan(trace_params, trace_data)

    assert trace.shape == (4, 1)
    assert np.allclose(expected, trace, equal_nan=True)
