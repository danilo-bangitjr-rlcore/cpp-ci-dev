from torch import Tensor

from corerl.component.buffer.buffers import PriorityBuffer, PriorityReplayBufferConfig
from corerl.data_pipeline.datatypes import NewTransition, NewTransitionBatch, Step, StepBatch

def test_sample_mini_batch():
    cfg = PriorityReplayBufferConfig(seed=0, memory=5, batch_size=2)
    buffer = PriorityBuffer(cfg)

    step_1 = Step(1.0, Tensor([0.5]), 0.99, Tensor([0.2, 0.4, 0.6, 0.8]), True)
    step_2 = Step(0.9, Tensor([0.6]), 0.99, Tensor([0.3, 0.5, 0.7, 0.9]), True)
    step_3 = Step(0.8, Tensor([0.7]), 0.99, Tensor([0.4, 0.6, 0.8, 1.0]), True)

    trans_1 = NewTransition(step_1, step_2, 1)
    trans_2 = NewTransition(step_2, step_3, 1)
    trans_3 = NewTransition(step_1, step_3, 2)

    buffer.feed(trans_1)
    buffer.feed(trans_2)
    buffer.feed(trans_3)

    batch = buffer.sample()[0]

    # With seed=0 and priorities of [0.25, 0.25, 0.5], the sampled indices are [2, 1]
    expected = NewTransitionBatch(
        StepBatch(
            Tensor([[1.0], [0.9]]),
            Tensor([[0.5], [0.6]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7, 0.9]]),
            Tensor([[True], [True]]),
        ),
        StepBatch(
            Tensor([[0.8], [0.8]]),
            Tensor([[0.7], [0.7]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0]]),
            Tensor([[True], [True]]),
        ),
        Tensor([[2], [1]]),
    )

    assert batch == expected
