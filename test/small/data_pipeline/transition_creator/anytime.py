import numpy as np
import pandas as pd
import datetime
from torch import Tensor
import torch

from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.transition_creators.anytime import (
    AnytimeTransitionCreator,
    AnytimeTransitionCreatorConfig)


def test_anytime_1():
    state_col = [np.array([i]) for i in range(4)]
    cols = {"state": state_col, "action": [0, 0, 1, 1], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df)
    pf.action_tags = ['action']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 1
    t_0 = transitions[0]

    assert torch.equal(t_0.state, Tensor([0.]))
    assert torch.equal(t_0.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.n_step_reward == 1
    assert torch.equal(t_0.next_state, Tensor([1.]))
    assert not t_0.terminated
    assert not t_0.truncate


def test_anytime_2():
    state_col = [np.array([i]) for i in range(4)]
    cols = {"state": state_col, "action": [0, 0, 0, 1], "reward": [1, 1, 1, 1]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df)
    pf.action_tags = ['action']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 10
    cfg.gamma = 0.9
    cfg.n_step = 1

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 2
    t_0 = transitions[0]
    t_1 = transitions[1]

    assert torch.equal(t_0.state, Tensor([0.]))
    assert torch.equal(t_0.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.n_step_reward == 1.0
    assert torch.equal(t_0.next_state, Tensor([1.]))
    assert not t_0.terminated
    assert not t_0.truncate

    assert torch.equal(t_1.state, Tensor([1.]))
    assert torch.equal(t_1.action, Tensor([0.]))
    assert t_1.n_steps == 1
    assert t_1.n_step_reward == 1.0
    assert torch.equal(t_1.next_state, Tensor([2.]))
    assert not t_1.terminated
    assert not t_1.truncate


def test_anytime_3():
    state_col = [np.array([i]) for i in range(7)]
    cols = {"state": state_col, "action": [0, 0, 1, 1, 1, 2, 2], "reward": [1, 1, 1, 1, 1, 1, 1]}

    dates = [
        datetime.datetime(2024, 1, 1, 1, i) for i in range(7)
    ]

    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)
    pf = PipelineFrame(df)
    pf.action_tags = ['action']

    cfg = AnytimeTransitionCreatorConfig()
    cfg.steps_per_decision = 3
    cfg.gamma = 0.9
    cfg.n_step = None

    tc = AnytimeTransitionCreator(cfg)
    transitions, _ = tc._inner_call(pf, tc_ts=None)

    assert len(transitions) == 4
    t_0 = transitions[0]
    t_1 = transitions[1]
    t_2 = transitions[2]
    t_3 = transitions[3]

    assert torch.equal(t_0.state, Tensor([0.]))
    assert torch.equal(t_0.action, Tensor([0.]))
    assert t_0.n_steps == 1
    assert t_0.n_step_reward == 1.
    assert torch.equal(t_0.next_state, Tensor([1.]))
    assert not t_0.terminated
    assert not t_0.truncate

    assert torch.equal(t_1.state, Tensor([2.]))
    assert torch.equal(t_1.action, Tensor([1.]))
    assert t_1.n_steps == 3
    assert t_1.n_step_reward == 2.71
    assert torch.equal(t_1.next_state, Tensor([5.]))
    assert not t_1.terminated
    assert not t_1.truncate

    assert torch.equal(t_2.state, Tensor([3.]))
    assert torch.equal(t_2.action, Tensor([1.]))
    assert t_2.n_steps == 2
    assert t_2.n_step_reward == 1.9
    assert torch.equal(t_2.next_state, Tensor([5.]))
    assert not t_2.terminated
    assert not t_2.truncate

    assert torch.equal(t_3.state, Tensor([4.]))
    assert torch.equal(t_3.action, Tensor([1.]))
    assert t_3.n_steps == 1
    assert t_3.n_step_reward == 1.
    assert torch.equal(t_3.next_state, Tensor([5.]))
    assert not t_3.terminated
    assert not t_3.truncate



#
# def test_anytime_ts_1():
#     state_col = [np.array([i]) for i in range(4)]
#     cols = {"state": state_col, "action": [0, 0, 1, 1], "reward": [1, 1, 1, 1]}
#     dates = [
#         datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
#     ]
#     datetime_index = pd.DatetimeIndex(dates)
#     df = pd.DataFrame(cols, index=datetime_index)
#     pf = PipelineFrame(df)
#     pf.action_tags = ['action']
#
#     cfg = AnytimeTransitionCreatorConfig()
#     cfg.steps_per_decision = 10
#     cfg.gamma = 0.9
#     cfg.n_step = None
#
#     tc = AnytimeTransitionCreator(cfg)
#     transitions, tc_ts = tc._inner_call(pf, tc_ts=None)
#     assert tc_ts is not None
#
#     assert len(transitions) == 2
#     t_0 = transitions[0]
#     t_1 = transitions[1]
#
#     assert torch.equal(t_0.state, Tensor([0.]))
#     assert torch.equal(t_0.action, Tensor([0.]))
#     assert t_0.n_steps == 2
#     assert t_0.n_step_reward == 1.9
#     assert torch.equal(t_0.next_state, Tensor([2.]))
#     assert not t_0.terminated
#     assert not t_0.truncate
#
#     assert torch.equal(t_1.state, Tensor([1.]))
#     assert torch.equal(t_1.action, Tensor([0.]))
#     assert t_1.n_steps == 1
#     assert t_1.n_step_reward == 1.0
#     assert torch.equal(t_1.next_state, Tensor([2.]))
#     assert not t_1.terminated
#     assert not t_1.truncate
#
#     state_col = [np.array([i]) for i in range(4, 8)]
#     cols = {"state": state_col, "action": [1, 1, 0, 0], "reward": [1, 1, 1, 1]}
#     dates = [
#         datetime.datetime(2024, 1, 1, 1, i) for i in range(4, 8)
#     ]
#     datetime_index = pd.DatetimeIndex(dates)
#     df = pd.DataFrame(cols, index=datetime_index)
#     pf_2 = PipelineFrame(df)
#     pf_2.action_tags = ['action']
#
#     transitions, tc_ts = tc._inner_call(pf_2, tc_ts=tc_ts)
#
#     assert len(transitions) == 4
#     t_0 = transitions[0]
#     t_1 = transitions[1]
#     t_2 = transitions[2]
#     t_3 = transitions[3]
#
#     assert torch.equal(t_0.state, Tensor([2.]))
#     assert torch.equal(t_0.action, Tensor([1.]))
#     assert t_0.n_steps == 4
#     assert t_0.n_step_reward == 3.439
#     assert torch.equal(t_0.next_state, Tensor([6.]))
#     assert not t_0.terminated
#     assert not t_0.truncate
#
#     assert torch.equal(t_1.state, Tensor([3.]))
#     assert torch.equal(t_1.action, Tensor([1.]))
#     assert t_1.n_steps == 3
#     assert t_1.n_step_reward == 2.71
#     assert torch.equal(t_1.next_state, Tensor([6.]))
#     assert not t_1.terminated
#     assert not t_1.truncate
#
#     assert torch.equal(t_2.state, Tensor([4.]))
#     assert torch.equal(t_2.action, Tensor([1.]))
#     assert t_2.n_steps == 2
#     assert t_2.n_step_reward == 1.9
#     assert torch.equal(t_2.next_state, Tensor([6.]))
#     assert not t_2.terminated
#     assert not t_2.truncate
#
#     assert torch.equal(t_3.state, Tensor([5.]))
#     assert torch.equal(t_3.action, Tensor([1.]))
#     assert t_3.n_steps == 1
#     assert t_3.n_step_reward == 1.
#     assert torch.equal(t_3.next_state, Tensor([6.]))
#     assert not t_3.terminated
#     assert not t_3.truncate
#
#
# def test_anytime_ts_2_data_gap():
#     state_col = [np.array([i]) for i in range(4)]
#     cols = {"state": state_col, "action": [0, 0, 1, 1], "reward": [1, 1, 1, 1]}
#     dates = [
#         datetime.datetime(2024, 1, 1, 1, i) for i in range(4)
#     ]
#     datetime_index = pd.DatetimeIndex(dates)
#     df = pd.DataFrame(cols, index=datetime_index)
#     pf = PipelineFrame(df)
#     pf.action_tags = ['action']
#     pf.data_gap = True  # NOTE: there is now a data gap
#
#     cfg = AnytimeTransitionCreatorConfig()
#     cfg.steps_per_decision = 10
#     cfg.gamma = 0.9
#     cfg.n_step = None
#
#     tc = AnytimeTransitionCreator(cfg)
#     transitions, tc_ts = tc._inner_call(pf, tc_ts=None)
#
#     assert len(transitions) == 2
#     t_0 = transitions[0]
#     t_1 = transitions[1]
#
#     assert torch.equal(t_0.state, Tensor([0.]))
#     assert torch.equal(t_0.action, Tensor([0.]))
#     assert t_0.n_steps == 2
#     assert t_0.n_step_reward == 1.9
#     assert torch.equal(t_0.next_state, Tensor([2.]))
#     assert not t_0.terminated
#     assert not t_0.truncate
#
#     assert torch.equal(t_1.state, Tensor([1.]))
#     assert torch.equal(t_1.action, Tensor([0.]))
#     assert t_1.n_steps == 1
#     assert t_1.n_step_reward == 1.0
#     assert torch.equal(t_1.next_state, Tensor([2.]))
#     assert not t_1.terminated
#     assert not t_1.truncate
#
#     state_col = [np.array([i]) for i in range(4, 8)]
#     cols = {"state": state_col, "action": [1, 1, 0, 0], "reward": [1, 1, 1, 1]}
#     dates = [
#         datetime.datetime(2024, 1, 1, 1, i) for i in range(5, 9)
#     ]
#     datetime_index = pd.DatetimeIndex(dates)
#     df = pd.DataFrame(cols, index=datetime_index)
#     pf_2 = PipelineFrame(df)
#     pf_2.action_tags = ['action']
#
#     transitions, tc_ts = tc._inner_call(pf_2, tc_ts=tc_ts)
#
#     assert len(transitions) == 3
#     t_0 = transitions[0]
#     t_1 = transitions[1]
#     t_2 = transitions[2]
#
#     assert torch.equal(t_0.state, Tensor([2.]))
#     assert torch.equal(t_0.action, Tensor([1.]))
#     assert t_0.n_steps == 1
#     assert t_0.n_step_reward == 1
#     assert torch.equal(t_0.next_state, Tensor([3.]))
#     assert not t_0.terminated
#     assert not t_0.truncate
#
#     # NOTE: NO transition from 3->4 because of the data gap
#
#     assert torch.equal(t_1.state, Tensor([4.]))
#     assert torch.equal(t_1.action, Tensor([1.]))
#     assert t_1.n_steps == 2
#     assert t_1.n_step_reward == 1.9
#     assert torch.equal(t_1.next_state, Tensor([6.]))
#     assert not t_1.terminated
#     assert not t_1.truncate
#
#     assert torch.equal(t_2.state, Tensor([5.]))
#     assert torch.equal(t_2.action, Tensor([1.]))
#     assert t_2.n_steps == 1
#     assert t_2.n_step_reward == 1
#     assert torch.equal(t_2.next_state, Tensor([6.]))
#     assert not t_2.terminated
#     assert not t_2.truncate
#
#
# def test_anytime_online_1():
#     """
#     Adds actions up to the steps per decision, which triggers creating transitions.
#     """
#     cfg = AnytimeTransitionCreatorConfig()
#     cfg.steps_per_decision = 3
#     cfg.gamma = 0.9
#     cfg.n_step = None
#
#     tc = AnytimeTransitionCreator(cfg)
#
#     tc_ts = None
#
#     transitions = []
#     for i in range(4):
#         state_col = [np.array([i])]
#         cols = {"state": state_col, "action": [1], "reward": [1]}
#         dates = [datetime.datetime(2024, 1, 1, 1, i)]
#         datetime_index = pd.DatetimeIndex(dates)
#         df = pd.DataFrame(cols, index=datetime_index)
#         pf = PipelineFrame(df)
#         pf.action_tags = ['action']
#
#         new_transitions, tc_ts = tc._inner_call(pf, tc_ts)
#         transitions += new_transitions
#
#         if i != 3:
#             assert len(transitions) == 0
#         else:
#             assert len(transitions) == 3
#
#     t_0 = transitions[0]
#     t_1 = transitions[1]
#     t_2 = transitions[2]
#
#     assert torch.equal(t_0.state, Tensor([0.]))
#     assert torch.equal(t_0.action, Tensor([1.]))
#     assert t_0.n_steps == 3
#     assert t_0.n_step_reward == 2.71
#     assert torch.equal(t_0.next_state, Tensor([3.]))
#     assert not t_0.terminated
#     assert not t_0.truncate
#
#     assert torch.equal(t_1.state, Tensor([1.]))
#     assert torch.equal(t_1.action, Tensor([1.]))
#     assert t_1.n_steps == 2
#     assert t_1.n_step_reward == 1.9
#     assert torch.equal(t_1.next_state, Tensor([3.]))
#     assert not t_1.terminated
#     assert not t_1.truncate
#
#     assert torch.equal(t_2.state, Tensor([2.]))
#     assert torch.equal(t_2.action, Tensor([1.]))
#     assert t_2.n_steps == 1
#     assert t_2.n_step_reward == 1.0
#     assert torch.equal(t_2.next_state, Tensor([3.]))
#     assert not t_2.terminated
#     assert not t_2.truncate
#
# def test_anytime_online_2():
#     """
#     Adds actions until a chage of action
#     """
#     cfg = AnytimeTransitionCreatorConfig()
#     cfg.steps_per_decision = 3
#     cfg.gamma = 0.9
#     cfg.n_step = None
#
#     tc = AnytimeTransitionCreator(cfg)
#
#     tc_ts = None
#
#     transitions = []
#     for i in range(4):
#         state_col = [np.array([i])]
#
#         if i == 3:
#             a = 2
#         else:
#             a = 1
#
#         cols = {"state": state_col, "action": [a], "reward": [1]}
#         dates = [datetime.datetime(2024, 1, 1, 1, i)]
#         datetime_index = pd.DatetimeIndex(dates)
#         df = pd.DataFrame(cols, index=datetime_index)
#         pf = PipelineFrame(df)
#         pf.action_tags = ['action']
#
#         new_transitions, tc_ts = tc._inner_call(pf, tc_ts)
#         transitions += new_transitions
#
#         if i != 3:
#             assert len(transitions) == 0
#         else:
#             assert len(transitions) == 2

    # t_0 = transitions[0]
    # t_1 = transitions[1]
    # t_2 = transitions[2]
    #
    # assert torch.equal(t_0.state, Tensor([0.]))
    # assert torch.equal(t_0.action, Tensor([1.]))
    # assert t_0.n_steps == 3
    # assert t_0.n_step_reward == 2.71
    # assert torch.equal(t_0.next_state, Tensor([3.]))
    # assert not t_0.terminated
    # assert not t_0.truncate
    #
    # assert torch.equal(t_1.state, Tensor([1.]))
    # assert torch.equal(t_1.action, Tensor([1.]))
    # assert t_1.n_steps == 2
    # assert t_1.n_step_reward == 1.9
    # assert torch.equal(t_1.next_state, Tensor([3.]))
    # assert not t_1.terminated
    # assert not t_1.truncate
    #
    # assert torch.equal(t_2.state, Tensor([2.]))
    # assert torch.equal(t_2.action, Tensor([1.]))
    # assert t_2.n_steps == 1
    # assert t_2.n_step_reward == 1.0
    # assert torch.equal(t_2.next_state, Tensor([3.]))
    # assert not t_2.terminated
    # assert not t_2.truncate