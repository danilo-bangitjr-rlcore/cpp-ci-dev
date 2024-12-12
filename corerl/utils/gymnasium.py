import gymnasium as gym

from corerl.data_pipeline.tag_config import TagConfig


def gen_tag_configs_from_env(env: gym.Env) -> dict[str, list[TagConfig]]:
    """
    Given a gymnasium.Env, generate the tag configuration for setting up an OPC experiment.
    Order of tags within the dict.values are important.
    """
    action_space: gym.Space = env.action_space
    assert isinstance(
        action_space, gym.spaces.Box
    ), f"only support Box type action_space, received {action_space}"
    assert len(
        action_space.shape
    ) == 1, f"only support action_space with dimensionality 1, received {action_space.shape}"
    n_actions = action_space.shape[0]

    observation_space: gym.Space = env.observation_space
    assert isinstance(
        observation_space, gym.spaces.Box
    ), f"only support Box type observation_space, received {observation_space}"
    assert len(
        observation_space.shape
    ) == 1, f"only support observation_space with dimensionality 1, received {observation_space.shape}"
    n_obs = observation_space.shape[0]

    tags = {
        'action': [],
        'observation': [],
        'meta': [],
    }

    for i in range(n_actions):
        tags['action'].append(TagConfig(
            name= f'action_{i}',
            bounds= (action_space.low[0].item() , action_space.high[0].item()),
            # outlier= None,
            # imputer= None,
            # reward_constructor= None,
            state_constructor= None,
            is_action= True
        ))

    for i in range(n_obs):
        tags['observation'].append(TagConfig(
            name = f'observation_{i}',
            bounds = (action_space.low[0].item(), action_space.high[0].item()),
            # outlier= None,
            # imputer= None,
            # reward_constructor= None,
            state_constructor= None,
            is_action = False
        ))

    # these are hardcoded
    tags['meta'].append(TagConfig(
        name = 'reward',
        # outlier= None,
        # imputer= None,
        # reward_constructor= None,
        state_constructor= None,
        is_action = False
    ))
    tags['meta'].append(TagConfig(
        name = 'terminated',
        # note: bounds is (None, None) but these are booleans
        # outlier= None,
        # imputer= None,
        # reward_constructor= None,
        state_constructor= None,
        is_action = False
    ))
    tags['meta'].append(TagConfig(
        name = 'truncated',
        # note: bounds is (None, None) but these are booleans
        # outlier= None,
        # imputer= None,
        # reward_constructor= None,
        state_constructor= None,
        is_action = False
    ))

    return tags
