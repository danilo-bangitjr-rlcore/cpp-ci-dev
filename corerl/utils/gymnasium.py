import gymnasium as gym

from corerl.data_pipeline.oddity_filters.identity import IdentityFilterConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import NullConfig, TraceConfig
from corerl.data_pipeline.transforms.identity import IdentityConfig


def gen_tag_configs_from_env(env: gym.Env, include_meta: bool = False) -> list[TagConfig]:
    """
    Given a gymnasium.Env, generate the tag configuration for setting up an OPC experiment.
    """
    action_space: gym.Space = env.action_space
    assert isinstance(action_space, gym.spaces.Box), f"only support Box type action_space, received {action_space}"
    assert (
        len(action_space.shape) == 1
    ), f"only support action_space with dimensionality 1, received {action_space.shape}"
    n_actions = action_space.shape[0]

    observation_space: gym.Space = env.observation_space
    assert isinstance(
        observation_space, gym.spaces.Box
    ), f"only support Box type observation_space, received {observation_space}"
    assert (
        len(observation_space.shape) == 1
    ), f"only support observation_space with dimensionality 1, received {observation_space.shape}"
    n_obs = observation_space.shape[0]

    tag_configs: list[TagConfig] = []

    for i in range(n_actions):
        tag_configs.append(
            TagConfig(
                name=f"action-{i}",
                operating_range=(action_space.low[i].item(), action_space.high[i].item()),
                state_constructor=[NullConfig()],
                action_constructor=[IdentityConfig()]
            )
        )

    for i in range(n_obs):
        tag_configs.append(
            TagConfig(
                name=f"tag-{i}",
                operating_range=(observation_space.low[i].item(), observation_space.high[i].item()),
                state_constructor=[
                    TraceConfig(
                        trace_values=[0.0]
                    )
                ],
            )
        )

    if include_meta:
        # these are hardcoded
        tag_configs.append(
            TagConfig(
                name="gym_reward",
                outlier=IdentityFilterConfig(),
                state_constructor=[],
                is_meta=True,
            )
        )
        tag_configs.append(
            TagConfig(
                name="terminated",
                outlier=IdentityFilterConfig(),
                is_meta=True,
            )
        )
        tag_configs.append(
            TagConfig(
                name="truncated",
                outlier=IdentityFilterConfig(),
                is_meta=True,
            )
        )

    return tag_configs
