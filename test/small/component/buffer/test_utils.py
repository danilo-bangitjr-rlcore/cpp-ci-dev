import copy

import numpy as np
import pytest
import os
import shutil

import pickle as pkl
from omegaconf import DictConfig
from pathlib import Path
from corerl.data_pipeline.datatypes import Transition
from corerl.component.buffer.buffers import EnsembleUniformBuffer
from corerl.component.buffer.utils import (load_pkl_buffer,
                                           subsampling_buffer)


@pytest.fixture
def cfg():
    cfg = DictConfig(
        {
            "memory": 11,
            "name": "uniform",
            "device": "cpu",
            "seed": 0,
            "ensemble": 1,
            "data_subset": 1.,
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
    test_path = "temp/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    test_path_critic = "{}/critic_buffer.pkl".format(test_path)
    test_path_actor = "{}/policy_buffer.pkl".format(test_path)
    with open(test_path_critic, "wb") as f:
        pkl.dump(buffer, f)
    with open(test_path_actor, "wb") as f:
        pkl.dump(buffer, f)
    load_pkl_buffer(path=Path(test_path), mode="all")
    load_pkl_buffer(path=Path(test_path), mode="random", seed=0, perc=1.)
    load_pkl_buffer(path=Path(test_path), mode="latest", perc=1.)
    shutil.rmtree(test_path)

def test_subsampling_buffer(buffer):
    b_len = buffer.size
    buffer_copy = copy.deepcopy(buffer)
    new_buffer = subsampling_buffer(buffer_copy, "all")
    assert new_buffer.size == b_len
    buffer_copy = copy.deepcopy(buffer)
    new_buffer = subsampling_buffer(buffer_copy, "random",
                                    seed=0, perc=1.)
    assert new_buffer.size == b_len
    buffer_copy = copy.deepcopy(buffer)
    new_buffer = subsampling_buffer(buffer_copy, "random",
                                    seed=0, perc=0.75)
    assert new_buffer.size[0] == int(b_len[0] * 0.75)
    buffer_copy = copy.deepcopy(buffer)
    new_buffer = subsampling_buffer(buffer_copy, "latest",
                                    seed=0, perc=0.75)
    assert new_buffer.size[0] == int(b_len[0] * 0.75)
