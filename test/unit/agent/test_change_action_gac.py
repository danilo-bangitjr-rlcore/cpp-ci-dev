import pytest
import numpy as np
import torch
from omegaconf import DictConfig

from corerl.agent.greedy_ac import GreedyACConfig, GreedyAC
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.network.networks import EnsembleCriticConfig, NNTorsoConfig
from corerl.component.policy.factory import BaseNNConfig
from corerl.component.optimizers.factory import OptimConfig
from corerl.component.buffer.buffers import EnsembleUniformBuffer
from corerl.data.data import Transition
from corerl.data.normalizer.action import IdentityNormalizerConfig


@pytest.fixture
def buffer_cfg():
    buffer_cfg = DictConfig(
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
    return buffer_cfg


@pytest.fixture
def actor_cfg(buffer_cfg):
    actor_cfg = NetworkActorConfig()
    actor_cfg.actor_network = BaseNNConfig()
    actor_cfg.actor_network.dist = 'beta'
    actor_cfg.actor_network.head_layer_init = 'Xavier'
    actor_cfg.actor_network.head_bias = True
    actor_cfg.actor_network.head_activation = [
        [{"name": "softplus"}, {"name": "bias", "args": [1]}],
        [
            {
                "name": "tanh_shift",
                "kwargs": {"shift": -4, "denom": 1, "high": 10000, "low": 1}
            }
        ]
    ]
    actor_cfg.actor_optimizer = OptimConfig()
    actor_cfg.actor_optimizer.name = 'adam'
    actor_cfg.buffer = buffer_cfg
    return actor_cfg


@pytest.fixture
def critic_cfg(buffer_cfg):
    critic_cfg = EnsembleCriticConfig()
    critic_cfg.critic_network = DictConfig(
        {
            'base': NNTorsoConfig(),
            'ensemble': 2,
            'name': 'ensemble',
            'reduct': 'min',
            'vmap': False,
        }
    )
    critic_cfg.polyak = 0.995
    critic_cfg.target_sync_freq = 1
    critic_cfg.critic_optimizer = DictConfig(
        {
            'name': 'adam',
            'lr': 0.0003,
            'weight_decay': 0.0
        }
    )
    critic_cfg.buffer = buffer_cfg
    return critic_cfg



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
            state=np.ones(3) * i,
            action=np.zeros(1),
            next_obs=np.ones(1) * (i+1),
            next_state=np.ones(3) * (i+1),
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


@pytest.fixture
def act_normalizer():
    return IdentityNormalizerConfig()

@pytest.fixture
def dummy_normalizer():
    class DummyNormalizer:
        def __init__(self):
            pass

        def __call__(self, x):
            return x

        def denormalize(self, x):
            return x

    return DummyNormalizer()

@pytest.fixture
def gac(actor_cfg, critic_cfg, buffer, dummy_normalizer):
    cfg = GreedyACConfig()
    cfg.delta_actor = True
    cfg.guardrail_low = np.zeros(1)
    cfg.guardrail_high = np.ones(1)
    cfg.actor = actor_cfg
    cfg.critic = critic_cfg
    cfg.buffer = buffer
    state_dim = 1
    action_dim = 1
    gac = GreedyAC(cfg, state_dim, action_dim)
    gac.set_normalizer(dummy_normalizer, dummy_normalizer)
    return gac


def test_delta_to_direct(gac):
    delta_action = torch.tensor(np.asarray([[0.1]]))
    prev_action = torch.tensor(np.asarray([[0.5]]))
    direct_action = gac.delta_to_direct(delta_action, prev_action)
    assert np.allclose(delta_action+prev_action, direct_action)

    delta_action = torch.tensor(np.asarray([[0.7]]))
    prev_action = torch.tensor(np.asarray([[0.5]]))
    direct_action = gac.delta_to_direct(delta_action, prev_action)
    assert np.allclose(torch.clamp(delta_action+prev_action, 0, 1), direct_action)

    delta_action = torch.tensor(np.asarray([[-0.7]]))
    prev_action = torch.tensor(np.asarray([[0.5]]))
    direct_action = gac.delta_to_direct(delta_action, prev_action)
    assert np.allclose(torch.clamp(delta_action+prev_action, 0, 1), direct_action)

def test_direct_to_delta(gac):
    direct_action = torch.tensor(np.asarray([[0.7]]))
    prev_action = torch.tensor(np.asarray([[0.5]]))
    delta_action = gac.direct_to_delta(direct_action, prev_action)
    assert np.allclose(direct_action - prev_action, delta_action)

def test_get_action_from_state(gac):
    state = np.arange(3).reshape(1, -1)
    assert gac.get_action_from_state(state) == state[0, 1]

    state = np.arange(3)
    assert gac.get_action_from_state(state) == state[1]

