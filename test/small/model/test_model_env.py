from torch import Tensor

from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NullConfig
from corerl.environment.model_env import ModelEnv, ModelEnvConfig
from corerl.models.dummy import DummyEndoModelConfig, DummyModelConfig, model_group
from corerl.utils.torch import tensor_allclose


def get_dummy_model_env(
        rollout_len: int,
        initial_states: list[Tensor],
        exo_seqs: list[list[Tensor]] | None = None,
) -> ModelEnv:
    cfg = ModelEnvConfig(rollout_len=rollout_len)
    env = ModelEnv(cfg)

    tags = [
        TagConfig(
            name='endo',
            is_endogenous=True,
        ),

        TagConfig(
            name='exo',
            is_endogenous=False,
        ),

        TagConfig(
            name='action',
            action_constructor=[],
            state_constructor=[NullConfig()]
        ),
    ]

    if exo_seqs is not None:
        model_cfg = DummyEndoModelConfig()
    else:
        model_cfg = DummyModelConfig()

    model = model_group.dispatch(model_cfg, tags)
    env.set_model(model)
    env.set_initial_states(
        initial_states,
        exo_seqs
    )
    return env


def test_init():
    initial_states = [
        Tensor([0.0, 0.0])
    ]
    _ = get_dummy_model_env(10, initial_states)


def test_rollout_counter():
    rollout_len = 10
    exp_initial_state = Tensor([0.0, 0.0])
    initial_states = [exp_initial_state]
    env = get_dummy_model_env(rollout_len, initial_states)
    initial_state, _ = env.reset()
    assert tensor_allclose(initial_state, exp_initial_state)

    action = Tensor([1.])
    for i in range(rollout_len):
        obs, reward, term, truncate, _ = env.step(action)
        assert tensor_allclose(obs, exp_initial_state)
        assert not term
        if i == rollout_len - 1:
            assert truncate
        else:
            assert not truncate


def test_exo_sequence():
    """
    Tests to see if exogenous observations are correctly concatenated to the observation.
    """
    rollout_len = 10
    exp_initial_state = Tensor([0.0, 0.0])
    initial_states = [exp_initial_state]
    exo_seqs = [
        [Tensor([i]) for i in range(rollout_len)]
    ]
    env = get_dummy_model_env(rollout_len, initial_states, exo_seqs)

    initial_state, _ = env.reset()
    assert tensor_allclose(initial_state, exp_initial_state)

    action = Tensor([1.])
    for i in range(rollout_len):
        obs, reward, term, truncate, _ = env.step(action)
        exp_obs = Tensor([0., i])
        assert tensor_allclose(obs, exp_obs)
