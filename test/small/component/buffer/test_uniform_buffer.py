from torch import Tensor

from corerl.component.buffer.uniform import UniformBuffer, UniformReplayBufferConfig
from corerl.data_pipeline.datatypes import DataMode, Step, StepBatch, Transition, TransitionBatch


def test_sample_mini_batch():
    cfg = UniformReplayBufferConfig(seed=0, memory=5, batch_size=2, combined=True)
    buffer = UniformBuffer(cfg)

    step_1 = Step(
        reward=1.0,
        action=Tensor([0.5]),
        gamma=0.99,
        state=Tensor([0.2, 0.4, 0.6, 0.8]),
        dp=True,
    )
    step_2 = Step(
        reward=0.9,
        action=Tensor([0.6]),
        gamma=0.99,
        state=Tensor([0.3, 0.5, 0.7, 0.9]),
        dp=True,
    )
    step_3 = Step(
        reward=0.8,
        action=Tensor([0.7]),
        gamma=0.99,
        state=Tensor([0.4, 0.6, 0.8, 1.0]),
        dp=True,
    )

    trans_1 = Transition(
        steps=[
            step_1,
            step_2,
        ],
        n_step_reward=0.9,
        n_step_gamma=0.99,
    )
    trans_2 = Transition(
        steps=[
            step_2,
            step_3,
        ],
        n_step_reward=0.8,
        n_step_gamma=0.99,
    )
    trans_3 = Transition(
        steps=[
            step_1,
            step_2,
            step_3,
        ],
        n_step_reward=0.9+0.99*0.8,
        n_step_gamma=0.99**2,
    )

    buffer.feed([trans_1], DataMode.OFFLINE)
    buffer.feed([trans_2], DataMode.OFFLINE)
    buffer.feed([trans_3], DataMode.OFFLINE)

    batch = buffer.sample()[0]

    # With seed=0, the sampled indices are [0, 1].
    # With combined=True, the first sampled index is replaced with the index of the last added transition.
    # So sampled indices becomes [2, 1], yielding the following TransitionBatch
    expected = TransitionBatch(
        # stub out the idxs as these are random and not meaningful
        batch.idxs,
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
        Tensor([[0.9+0.99*0.8], [0.8]]),
        Tensor([[0.99**2], [0.99]]),
    )

    assert batch == expected
