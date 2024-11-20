import numpy as np
import pytest
import os

import pickle as pkl
from omegaconf import DictConfig
from pathlib import Path
from corerl.data.data import Transition
from corerl.component.buffer.buffers import EnsembleUniformBuffer
from corerl.component.buffer.utils import (load_pkl_buffer,
                                           subsampling_buffer,
                                           subsampling_transitions,
                                           get_loaded_transitions_idx)


@pytest.fixture
def cfg():
    cfg = DictConfig(
        {
            "memory": 100000,
            "name": "uniform",
            "device": "cpu",
            "seed": 0,
            "ensemble": 1,
            "data_subset": 1.0,
            "batch_size": 1,
            "combined": True
        }
    )
    return cfg

@pytest.fixture
def data():
    data = []
    for i in range(10):
        trans = Transition(
            obs=np.ones(1) * i,
            state=np.ones(1) * i,
            action=np.zeros(1),
            next_obs=np.ones(1) * (i+1),
            next_state=np.ones(1) * (i+1),
            reward=0,
            n_step_reward=1,
        )
        data.append(trans)
    return data

@pytest.fixture
def buffer(cfg, data):
    buffer = EnsembleUniformBuffer(cfg)
    buffer.load(data)
    return buffer

def test_load_pkl_buffer(buffer):
    test_path = "temp/test_buffer_loading"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    test_path_critic = "{}/critic_buffer.pkl".format(test_path)
    test_path_actor = "{}/policy_buffer.pkl".format(test_path)
    with open(test_path_critic, "wb") as f:
        pkl.dump(buffer, f)
    with open(test_path_actor, "wb") as f:
        pkl.dump(buffer, f)
    assert load_pkl_buffer(path=Path(test_path), mode="all")
    assert load_pkl_buffer(path=Path(test_path), mode="random",
                           seed=0, perc=1.)
    assert load_pkl_buffer(path=Path(test_path), mode="latest",
                           perc=1.)