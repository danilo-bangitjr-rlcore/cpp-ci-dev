import gymnasium as gym
from corerl.tags.meta import MetaTagConfig
from corerl.tags.setpoint import SetpointTagConfig
from corerl.tags.tag_config import BasicTagConfig, TagConfig


def gen_tag_configs_from_env(env: gym.Env, include_meta: bool = False):
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
        observation_space, gym.spaces.Box,
    ), f"only support Box type observation_space, received {observation_space}"
    assert (
        len(observation_space.shape) == 1
    ), f"only support observation_space with dimensionality 1, received {observation_space.shape}"
    n_obs = observation_space.shape[0]

    tag_configs: list[TagConfig] = [
        SetpointTagConfig(
            name=f"action-{i}",
            operating_range=(action_space.low[0].item(), action_space.high[0].item()),
        )
        for i in range(n_actions)
    ]

    tag_configs.extend(
        BasicTagConfig(
            name=f"tag-{i}",
            operating_range=(observation_space.low[i].item(), observation_space.high[i].item()),
        )
        for i in range(n_obs)
    )

    if include_meta:
        # these are hardcoded
        tag_configs.append(
            MetaTagConfig(
                name="gym_reward",
            ),
        )
        tag_configs.append(
            MetaTagConfig(
                name="terminated",
            ),
        )
        tag_configs.append(
            MetaTagConfig(
                name="truncated",
            ),
        )

    return tag_configs
