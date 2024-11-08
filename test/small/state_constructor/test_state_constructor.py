import numpy as np
from corerl.state_constructor.examples import MultiTrace, MultiTraceConfig
from corerl.state_constructor.factory import init_state_constructor

def test_multitrace1():
    """
    Calling the init_sc factory function
    with a MultiTraceConfig gives back
    a MultiTrace state constructor.
    """
    cfg = MultiTraceConfig(
        trace_values=[0.9, 0.1],
    )

    sc = init_state_constructor(cfg)

    assert isinstance(sc, MultiTrace)


def test_multitrace2():
    """
    MultiTrace state constructor generates a trace
    with decay rate 0.1
    """
    cfg = MultiTraceConfig(
        trace_values=[0.1],
    )

    sc = init_state_constructor(cfg)

    x = np.ones(2)
    a = np.array([2.])
    got = sc(x, a)

    assert np.allclose(
        got,
        np.array([
            # whether this is an initial state
            # since we did not tell the sc that it is
            # assume no
            0.,
            # the action concatenated with the obs vec
            2., 1, 1,
            # the trace of the [act, obs] features
            1.8, 0.9, 0.9,
        ]),
    )
