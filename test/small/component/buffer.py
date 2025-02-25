import torch
from pyparsing import deque

from corerl.component.buffer import MixedHistoryBuffer, MixedHistoryBufferConfig
from corerl.data_pipeline.datatypes import DataMode, Transition
from corerl.state import AppState
from test.small.data_pipeline.test_transition_filter import make_test_step


def make_test_transition(start: int, len: int) -> Transition:
    steps = [make_test_step(start + i) for i in range(len+1)]
    transition = Transition(steps, 1, .99)
    return transition

def make_test_transitions(start:int, num: int, len: int) -> list[Transition]:
    transitions = []
    for i in range(start, start+num):
        transitions.append(make_test_transition(i, len))
    return transitions

def test_feed_online_mode(dummy_app_state: AppState):
    buffer_cfg = MixedHistoryBufferConfig(
        seed=42,
        memory=100,
        batch_size=10,
        n_most_recent=2,
    )

    buffer = MixedHistoryBuffer(buffer_cfg, dummy_app_state)
    online_transitions = make_test_transitions(0, 5, 1)
    idxs = buffer.feed(online_transitions, DataMode.ONLINE)
    assert len(idxs) == 5
    assert buffer._most_recent_online_idxs == deque([4, 3], maxlen=2)

    # now feed some transitions with a different mode
    offline_transitions = make_test_transitions(5, 5, 1)
    idxs = buffer.feed(offline_transitions, DataMode.OFFLINE)

    # most recent online idxs should not change
    assert buffer._most_recent_online_idxs == deque([4, 3], maxlen=2)


    samples = buffer.sample()

    for batch in samples:
        assert (batch.prior.state[:2, 0] == torch.Tensor([4, 3])).all()
