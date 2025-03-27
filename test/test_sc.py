import numpy as np
from src.interaction.state_constructor import StateConstructor

def test_normalize_only():
    obs_space = {
        'low': np.array([-1.0, 0.0]),
        'high': np.array([1.0, 2.0])
    }
    sc = StateConstructor(observation_space_info=obs_space)
    
    observation = {
        'x': np.array([0.0, 1.0])
    }
    
    result, state = sc(observation)
    expected = {
        'x': np.array([0.5, 0.5])  # normalized to [0,1] range
    }
    
    assert state is None
    assert len(result) == len(expected)
    np.testing.assert_array_almost_equal(result['x'], expected['x'])
    assert sc.get_state_dim() == 2

def test_trace_only():
    decays = [0.5, 0.9]
    sc = StateConstructor(trace_values=decays)
    
    observation = {
        'x': np.array([1.0, 2.0])
    }
    
    result, _ = sc(observation)
    
    assert len(result) == 3  # original + 2 traces
    assert 'x' in result
    assert 'x_trace-0.5' in result
    assert 'x_trace-0.9' in result
    np.testing.assert_array_equal(result['x'], observation['x'])
    
    array_state = sc.to_array(result)
    assert array_state.shape == (2, 3)  # (input_dim, original + 2 traces)

def test_normalize_and_trace():
    obs_space = {
        'low': np.array([-1.0]),
        'high': np.array([1.0])
    }
    decays = [0.5]
    sc = StateConstructor(observation_space_info=obs_space, trace_values=decays)
    
    observation = {
        'x': np.array([0.0])
    }
    
    result, _ = sc(observation)
    
    assert len(result) == 2  # original + 1 trace
    assert 'x' in result
    assert 'x_trace-0.5' in result
    np.testing.assert_array_almost_equal(result['x'], np.array([0.5]))  # normalized
    assert sc.get_state_dim() == 2  # 1 original + 1 trace
