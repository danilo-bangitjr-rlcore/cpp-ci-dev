from main import *
from omegaconf import DictConfig

from corerl.data_loaders.utils import make_anytime_trajectories
from corerl.calibration_models.factory import init_calibration_model
from corerl.data import Transition, Trajectory


def trajectories_to_transitions(trajectories: list[Trajectory]) -> list[Transition]:
    transitions = []
    for t in trajectories:
        transitions += t.transitions
    return transitions


def load_cm_offline_data_from_csv(cfg: DictConfig) -> dict:
    env = init_environment(cfg.env)
    dl = init_data_loader(cfg.data_loader)

    output_path = Path(cfg.offline_data.output_path)

    create_df = lambda dl_, filenames: dl_.load_data(filenames)
    all_data_df = load_or_create(output_path, [cfg.data_loader],
                                 'all_data_df', create_df, [dl, dl.all_filenames])
    train_data_df = load_or_create(output_path, [cfg.data_loader],
                                   'train_data_df', create_df, [dl, dl.train_filenames])
    test_data_df = load_or_create(output_path, [cfg.data_loader],
                                  'test_data_df', create_df, [dl, dl.test_filenames])

    assert not all_data_df.isnull().values.any()
    assert not train_data_df.isnull().values.any()
    assert not test_data_df.isnull().values.any()

    create_bounds = lambda dl_, df: dl.get_obs_max_min(df)
    obs_bounds = load_or_create(output_path, [cfg.data_loader],
                                'obs_bounds', create_bounds, [dl, all_data_df])

    env.observation_space = spaces.Box(low=obs_bounds[0], high=obs_bounds[1], dtype=np.float32)

    # this is the state constructor for the calibration model
    sc_cm = init_state_constructor(cfg.calibration_model.state_constructor, env)

    reward_func = init_reward_function(cfg.env.reward)
    create_obs_transitions = lambda dl_, r_func, df: dl_.create_obs_transitions(df, reward_func)
    train_obs_transitions = load_or_create(output_path, [cfg.data_loader],
                                           'train_obs_transitions', create_obs_transitions,
                                           [dl, reward_func, train_data_df])

    if test_data_df is not None:
        test_obs_transitions = load_or_create(output_path, [cfg.data_loader],
                                              'test_obs_transitions', create_obs_transitions,
                                              [dl, reward_func, test_data_df])
    else:
        test_obs_transitions = None

    # this is an interaction that uses the state constructor of the calibration model
    interaction_cm = init_interaction(cfg.interaction, env, sc_cm)
    create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_trajectories(
        obs_transitions,
        interaction_,
        sc_warmup=warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )

    # next, we will create the training and test transitions for the calibration model
    train_trajectories_cm = load_or_create(output_path,
                                           [cfg.data_loader, cfg.calibration_model.state_constructor, cfg.interaction],
                                           'train_trajectories_cm', create_trajectories,
                                           [train_obs_transitions, interaction_cm,
                                            cfg.calibration_model.state_constructor.warmup, True])

    if test_obs_transitions is not None:
        test_trajectories_cm = load_or_create(output_path,
                                              [cfg.data_loader, cfg.calibration_model.state_constructor,
                                               cfg.interaction],
                                              'test_trajectories_cm', create_trajectories,
                                              [test_obs_transitions, interaction_cm,
                                               cfg.calibration_model.state_constructor.warmup, True])
    else:
        train_trajectories_cm, test_trajectories_cm = train_trajectories_cm[0].split_at(3999)
        train_trajectories_cm = [train_trajectories_cm]
        test_trajectories_cm = [test_trajectories_cm]

    # finally, create the train and test transitions for the agent
    sc_agent = init_state_constructor(cfg.state_constructor, env)
    interaction_agent = init_interaction(cfg.interaction, env, sc_agent)

    train_trajectories_agent = load_or_create(output_path,
                                              [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                              'train_trajectories_agent', create_trajectories,
                                              [train_obs_transitions, interaction_agent, cfg.state_constructor.warmup,
                                               True])

    if test_obs_transitions is not None:
        test_trajectories_agent = load_or_create(output_path,
                                                 [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                                 'test_trajectories_agent', create_trajectories,
                                                 [test_obs_transitions, interaction_agent, cfg.state_constructor.warmup,
                                                  True])
    else:
        # TODO: get this index from somewhere else
        train_trajectories_agent, test_trajectories_agent = train_trajectories_agent[0].split_at(3999)
        train_trajectories_agent = [train_trajectories_agent]
        test_trajectories_agent = [test_trajectories_agent]

    return_dict = {
        'env': env,
        'sc_cm': sc_cm,
        'agent_sc': sc_agent,
        'interaction_cm': interaction_cm,
        'interaction_agent': interaction_agent,
        'train_trajectories_cm': train_trajectories_cm,
        'train_trajectories_agent': train_trajectories_agent,
        'test_trajectories_cm': test_trajectories_cm,
        'test_trajectories_agent': test_trajectories_agent,
        'train_transitions_cm': trajectories_to_transitions(train_trajectories_cm),
        'train_transitions_agent': trajectories_to_transitions(train_trajectories_agent),
        'test_transitions_cm': trajectories_to_transitions(test_trajectories_cm),
        'test_transitions_agent': trajectories_to_transitions(test_trajectories_agent)
    }

    return return_dict


def load_cm_offline_data_from_transitions(cfg: DictConfig) -> dict:
    output_path = Path(cfg.offline_data.output_path)
    env = init_environment(cfg.env)

    # We assume that transitions have been created with make_offline_transitions.py
    nothing_fn = lambda *args: None

    train_obs_transitions = load_or_create(output_path,
                                           [cfg.env, cfg.interaction, cfg.agent],
                                           'obs_transitions', nothing_fn,
                                           [])

    sc_cm = init_state_constructor(cfg.calibration_model.state_constructor, env)

    # this is an interaction that uses the state constructor of the calibration model
    interaction_cm = init_interaction(cfg.interaction, env, sc_cm)
    create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_trajectories(
        obs_transitions,
        interaction_,
        sc_warmup=warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )

    # next, we will create the training and test transitions for the calibration model
    train_trajectories_cm = load_or_create(output_path,
                                           [cfg.data_loader, cfg.calibration_model.state_constructor, cfg.interaction],
                                           'train_trajectories_cm', create_trajectories,
                                           [train_obs_transitions, interaction_cm,
                                            cfg.calibration_model.state_constructor.warmup, True])

    # TODO: get this index from somewhere else
    train_trajectories_cm, test_trajectories_cm = train_trajectories_cm[0].split_at(3999)
    train_trajectories_cm = [train_trajectories_cm]
    test_trajectories_cm = [test_trajectories_cm]

    # finally, create the train and test transitions for the agent
    sc_agent = init_state_constructor(cfg.state_constructor, env)
    interaction_agent = init_interaction(cfg.interaction, env, sc_agent)

    train_trajectories_agent = load_or_create(output_path,
                                              [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                              'train_trajectories_agent', create_trajectories,
                                              [train_obs_transitions, interaction_agent, cfg.state_constructor.warmup,
                                               True])

    # TODO: get this index from somewhere else
    train_trajectories_agent, test_trajectories_agent = train_trajectories_agent[0].split_at(3999)
    train_trajectories_agent = [train_trajectories_agent]
    test_trajectories_agent = [test_trajectories_agent]

    return_dict = {
        'env': env,
        'sc_cm': sc_cm,
        'agent_sc': sc_agent,
        'interaction_cm': interaction_cm,
        'interaction_agent': interaction_agent,
        'train_trajectories_cm': train_trajectories_cm,
        'train_trajectories_agent': train_trajectories_agent,
        'test_trajectories_cm': test_trajectories_cm,
        'test_trajectories_agent': test_trajectories_agent,
        'train_transitions_cm': trajectories_to_transitions(train_trajectories_cm),
        'train_transitions_agent': trajectories_to_transitions(train_trajectories_agent),
        'test_transitions_cm': trajectories_to_transitions(test_trajectories_cm),
        'test_transitions_agent': trajectories_to_transitions(test_trajectories_agent)
    }

    return return_dict


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    save_path = prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    init_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if cfg.offline_data.load_from == 'csv':
        train_info = load_cm_offline_data_from_csv(cfg)
    elif cfg.offline_data.load_from == 'transition':
        train_info = load_cm_offline_data_from_transitions(cfg)
    else:
        raise NotImplementedError

    reward_func = init_reward_function(cfg.env.reward)

    train_info["reward_func"] = reward_func
    cm = init_calibration_model(cfg.calibration_model, train_info)
    cm.train()

    env = train_info['env']
    agent_sc = train_info['agent_sc']

    state_dim, action_dim = get_state_action_dim(env, agent_sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)

    # agent should be pretty bad
    returns = cm.do_agent_rollouts(agent, train_info['test_trajectories_agent'], plot=True)
    print("returns", returns)

    # now, train the agent, is it better?
    offline_eval = offline_training(cfg, agent, train_info["train_transitions_agent"],
                                    train_info["test_transitions_agent"],,

    returns = cm.do_agent_rollouts(agent, train_info['test_trajectories_agent'], plot=True)
    print("returns", returns)


if __name__ == "__main__":
    main()
