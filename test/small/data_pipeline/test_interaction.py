from typing import Any
import numpy as np
import pandas as pd
import datetime
from torch import tensor

from corerl.data_pipeline.imputers.linear import LinearImputerConfig
from corerl.data_pipeline.interaction import InteractionWrapper
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transforms.trace import TraceConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.datatypes import Step, CallerCode, NewTransition
from corerl.data_pipeline.transition_creators.anytime import AnytimeTransitionCreatorConfig


def test_pipeline1():
    # -----------
    # -- Setup --
    # -----------
    cfg = PipelineConfig(
        tags=[
            TagConfig(
                name='tag-1',
                state_constructor=[],
            ),
            TagConfig(
                name='tag-2',
                bounds=(None, 10),
                imputer=LinearImputerConfig(max_gap=2),
                state_constructor=[
                    NormalizerConfig(),
                    TraceConfig(trace_values=[0.1]),
                ],
            ),
            TagConfig(name='action-1', is_action=True),
        ],
        agent_transition_creator=AnytimeTransitionCreatorConfig(
            steps_per_decision=2,
            gamma=0.9,
            n_step=None,
            only_dp_transitions=False,
        ),
        obs_interval_minutes=5,
    )

    start = datetime.datetime.now(datetime.UTC)
    Δ = datetime.timedelta(minutes=5)

    dates = [start + i * Δ for i in range(7)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-1', 'tag-2', 'reward', 'action-1']
    df = pd.DataFrame(
        data=[
            [np.nan, 0,        0,    0],
            [0,      2,        3,    1],
            [1,      4,        0,    0],
            [2,      6,        0,    1],
            [np.nan, np.nan,   0,    0],
            [4,      10,       1,    1],
            [5,      10,       0,    0],
        ],
        columns=cols,
        index=idx,
    )

    pipeline = Pipeline(cfg)

    # -----------------------------
    # -- Probe system-under-test --
    # -----------------------------
    interaction = InteractionWrapper(
        pipeline,
        action_period=datetime.timedelta(minutes=15),
        tol=datetime.timedelta(minutes=5),
    )

    got = pipeline(
        df,
        caller_code=CallerCode.ONLINE,
    )

    # ----------------------------
    # -- Check setup conditions --
    # ----------------------------
    assert got.transitions is not None and len(got.transitions) > 0
    assert got.transitions[-1] == NewTransition(
        prior=Step(reward=0, action=tensor([0.]), gamma=0.9, state=tensor([1.0, 0.378, 0., 1.]), dp=True),
        post=Step(reward=0, action=tensor([1.]), gamma=0.9, state=tensor([2.0, 0.5778, 1., 0.]), dp=False),
        n_steps=1,
    )


    # ------------------------
    # -- Check SUT outcomes --
    # ------------------------
    state = interaction.get_latest_state()
    assert state is not None

    # notice this state does not correspond to the last complete transition
    # instead it corresponds with the last complete _state_
    assert np.allclose(state, [0, 0, 5, 0.9977778])
