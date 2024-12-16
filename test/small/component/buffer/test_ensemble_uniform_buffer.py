from torch import Tensor

from corerl.component.buffer.buffers import EnsembleUniformBuffer, EnsembleUniformReplayBufferConfig
from corerl.data_pipeline.datatypes import NewTransition, NewTransitionBatch, Step, StepBatch

def test_sample_mini_batch():
    cfg = EnsembleUniformReplayBufferConfig(seed=0, memory=5, batch_size=2, combined=True, ensemble=2, data_subset=0.6)
    buffer = EnsembleUniformBuffer(cfg)

    step_1 = Step(1.0, Tensor([0.5]), 0.99, Tensor([0.2, 0.4, 0.6, 0.8]), True)
    step_2 = Step(0.9, Tensor([0.6]), 0.99, Tensor([0.3, 0.5, 0.7, 0.9]), True)
    step_3 = Step(0.8, Tensor([0.7]), 0.99, Tensor([0.4, 0.6, 0.8, 1.0]), True)

    trans_1 = NewTransition(step_1, step_2, 1)
    trans_2 = NewTransition(step_2, step_3, 1)
    trans_3 = NewTransition(step_1, step_3, 2)

    # With seed=0 and data_subset=0.6, the first buffer in the ensemble is fed trans_1, trans_3
    # and the second buffer in the ensemble is fed trans_2
    buffer.feed(trans_1)
    buffer.feed(trans_2)
    buffer.feed(trans_3)

    ensemble_batch = buffer.sample()
    assert len(ensemble_batch) == 2
    batch_1 = ensemble_batch[0]
    batch_2 = ensemble_batch[1]

    # With seed=0 and combined=True, the first buffer of the ensemble's sampled indices are [1, 1]
    expected_1 = NewTransitionBatch(
        StepBatch(
            Tensor([[1.0], [1.0]]),
            Tensor([[0.5], [0.5]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]]),
            Tensor([[True], [True]]),
        ),
        StepBatch(
            Tensor([[0.8], [0.8]]),
            Tensor([[0.7], [0.7]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0]]),
            Tensor([[True], [True]]),
        ),
        Tensor([[2], [2]]),
    )

    # The second buffer of the ensemble's sampled indices are [0, 0]
    expected_2 = NewTransitionBatch(
        StepBatch(
            Tensor([[0.9], [0.9]]),
            Tensor([[0.6], [0.6]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.3, 0.5, 0.7, 0.9], [0.3, 0.5, 0.7, 0.9]]),
            Tensor([[True], [True]]),
        ),
        StepBatch(
            Tensor([[0.8], [0.8]]),
            Tensor([[0.7], [0.7]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0]]),
            Tensor([[True], [True]]),
        ),
        Tensor([[1], [1]]),
    )

    assert batch_1 == expected_1
    assert batch_2 == expected_2
