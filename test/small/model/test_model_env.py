from torch import Tensor

from corerl.environment.model_env import ModelEnv, ModelEnvConfig
from corerl.models.dummy import DummyModel, DummyModelConfig
from corerl.utils.torch import tensor_allclose


def get_dummy_model_env(
        rollout_len: int,
        initial_states: list[Tensor]
) -> ModelEnv:
    cfg = ModelEnvConfig(rollout_len=rollout_len)
    env = ModelEnv(cfg)

    model_cfg = DummyModelConfig()
    model = DummyModel(model_cfg)

    env.set_model(model)
    env.set_initial_states(
        initial_states
    )
    return env


def test_init():
    initial_states = [Tensor([0.0])]
    _ = get_dummy_model_env(10, initial_states)


def test_rollout_counter():
    rollout_len = 10
    exp_initial_state = Tensor([0.0])
    initial_states = [exp_initial_state]
    env = get_dummy_model_env(rollout_len, initial_states)
    initial_state, _ = env.reset()
    assert tensor_allclose(initial_state, exp_initial_state)

    action = Tensor([1.])
    for i in range(rollout_len):
        obs, reward, term, truncate, _ = env.step(action)
        assert tensor_allclose(obs, exp_initial_state)
        assert not truncate
        if i == rollout_len - 1:
            assert term
        else:
            assert not term
