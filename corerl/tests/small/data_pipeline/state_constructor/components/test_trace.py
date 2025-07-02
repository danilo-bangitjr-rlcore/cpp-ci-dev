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

    new_carry, new_ts = trace_sc(carry, ts)

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
