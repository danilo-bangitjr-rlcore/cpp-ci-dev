import numpy as np

from src.interaction.state_constructor import StateConstructor


def test_normalize_only():
    obs_space = {
        'low': np.array([-1.0, 0.0]),
        'high': np.array([1.0, 2.0])
    }
    sc = StateConstructor(observation_space_info=obs_space)

    observation = {
        '0': np.array([0.0]),
        '1': np.array([1.0])
    }

    result, state = sc(observation)
    expected = {
        '0': np.array([0.5]),
        '1': np.array([0.5])
    }

    assert state is None
    assert len(result) == len(expected)
    np.testing.assert_array_almost_equal(result['0'], expected['0'])
    assert sc.get_state_dim() == 2

def test_trace_only():
    decays = [0.5, 0.9]
    sc = StateConstructor(trace_values=decays)

    observation = {
        '0': np.array([1.0, 2.0])
    }

    result, _ = sc(observation)

    assert len(result) == 3  # original + 2 traces
    assert '0' in result
    assert '0_trace-0.5' in result
    assert '0_trace-0.9' in result
    np.testing.assert_array_equal(result['0'], observation['0'])

    array_state = sc.to_array(result)
    assert array_state.shape == (2, 3)  # (input_dim, original + 2 traces)
