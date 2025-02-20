from torch import Tensor

from corerl.component.buffer.ensemble import EnsembleUniformBuffer, EnsembleUniformReplayBufferConfig
from corerl.data_pipeline.datatypes import DataMode, Step, StepBatch, Transition, TransitionBatch
from corerl.state import AppState


def test_sample_mini_batch(dummy_app_state: AppState):
    cfg = EnsembleUniformReplayBufferConfig(seed=0, memory=5, batch_size=2,
                                            n_most_recent=1, ensemble=2, data_subset=0.6)
    buffer = EnsembleUniformBuffer(cfg, dummy_app_state)

    step_1 = Step(
        reward=1.0,
        action=Tensor([0.5]),
        gamma=0.99,
        state=Tensor([0.2, 0.4, 0.6, 0.8]),
        dp=True,
        ac=False
    )
    step_2 = Step(
        reward=0.9,
        action=Tensor([0.6]),
        gamma=0.99,
        state=Tensor([0.3, 0.5, 0.7, 0.9]),
        dp=True,
        ac=True
    )
    step_3 = Step(
        reward=0.8,
        action=Tensor([0.7]),
        gamma=0.99,
        state=Tensor([0.4, 0.6, 0.8, 1.0]),
        dp=True,
        ac=True
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
        n_step_reward=0.9 + 0.99 * 0.8,
        n_step_gamma=0.99 ** 2,
    )

    # With seed=0 and data_subset=0.6, the first buffer in the ensemble is fed trans_1, trans_3
    # and the second buffer in the ensemble is fed trans_2
    buffer.feed([trans_1], DataMode.OFFLINE)
    buffer.feed([trans_2], DataMode.OFFLINE)
    buffer.feed([trans_3], DataMode.OFFLINE)

    ensemble_batch = buffer.sample()
    assert len(ensemble_batch) == 2
    batch_1 = ensemble_batch[0]
    batch_2 = ensemble_batch[1]

    # With seed=0 and combined=True, the first buffer of the ensemble's sampled indices are [2, 1]
    expected_1 = TransitionBatch(
        # stub out the idxs as these are random and not meaningful
        batch_1.idxs,
        StepBatch(
            reward=Tensor([[1.0], [0.9]]),
            action=Tensor([[0.5], [0.6]]),
            gamma=Tensor([[0.99], [0.99]]),
            state=Tensor([[0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7, 0.9]]),
            dp=Tensor([[True], [True]]),
            ac=Tensor([[False, True]]),
        ),
        StepBatch(
            Tensor([[0.8], [0.8]]),
            Tensor([[0.7], [0.7]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0]]),
            Tensor([[True], [True]]),
            ac=Tensor([[True, True]]),
        ),
        Tensor([[0.9+0.99*0.8], [0.8]]),
        Tensor([[0.99**2], [0.99]]),
    )

    # The second buffer of the ensemble's sampled indices are [2, 1] (are these actually gettind different data?)
    expected_2 = TransitionBatch(
        # stub out the idxs as these are random and not meaningful
        batch_2.idxs,
        StepBatch(
            Tensor([[1.0], [0.9]]),
            Tensor([[0.5], [0.6]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7, 0.9]]),
            Tensor([[True], [True]]),
            ac=Tensor([[False, True]]),
        ),
        StepBatch(
            Tensor([[0.8], [0.8]]),
            Tensor([[0.7], [0.7]]),
            Tensor([[0.99], [0.99]]),
            Tensor([[0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0]]),
            Tensor([[True], [True]]),
            ac=Tensor([[True, True]]),
        ),
        Tensor([[0.9+0.99*0.8], [0.8]]),
        Tensor([[0.99**2], [0.99]]),
    )

    assert batch_1 == expected_1
    assert batch_2 == expected_2
