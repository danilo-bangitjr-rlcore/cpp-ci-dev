from main import *
from corerl.data_loaders.utils import make_anytime_trajectories
from corerl.calibration_models.factory import init_calibration_model


def load_cm_offline_data_from_csv(cfg):
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


    print(all_data_df)
    print(train_data_df)
    print(test_data_df)

    create_bounds = lambda dl_, df: dl.get_obs_max_min(df)
    obs_bounds = load_or_create(output_path, [cfg.data_loader],
                                'obs_bounds', create_bounds, [dl, all_data_df])

    env.observation_space = spaces.Box(low=obs_bounds[0], high=obs_bounds[1], dtype=np.float32)
    sc = init_state_constructor(cfg.state_constructor, env)

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

    interaction = init_interaction(cfg.interaction, env, sc)
    create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_trajectories(
        obs_transitions,
        interaction_,
        sc_warmup=cfg.state_constructor.warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )
    train_trajectories = load_or_create(output_path,
                                        [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                        'train_trajectories', create_trajectories,
                                        [train_obs_transitions, interaction, cfg.state_constructor.warmup, True])

    if test_obs_transitions is not None:
        test_trajectories = load_or_create(output_path,
                                           [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                           'test_trajectories', create_trajectories,
                                           [test_obs_transitions, interaction, cfg.state_constructor.warmup,
                                            True])
    else:
        test_trajectories = None



    return env, sc, interaction, train_trajectories, test_trajectories


def load_cm_offline_data_from_transitions(cfg):
    output_path = Path(cfg.offline_data.output_path)

    # We assume that transitions have been created with make_offline_transitions.py
    nothing_fn = lambda *args: None

    train_obs_transitions = load_or_create(output_path,
                                           [cfg.env, cfg.state_constructor, cfg.interaction, cfg.agent],
                                           'obs_transitions', nothing_fn,
                                           [])

    # transitions = load_or_create(output_path,
    #                              [cfg.env, cfg.state_constructor, cfg.interaction, cfg.agent],
    #                              'transitions', nothing_fn,
    #                              [])

    create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_trajectories(
        obs_transitions,
        interaction_,
        sc_warmup=cfg.state_constructor.warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )

    env = init_environment(cfg.env)
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)

    train_trajectories = load_or_create(output_path,
                                        [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                        'train_trajectories', create_trajectories,
                                        [train_obs_transitions, interaction, cfg.state_constructor.warmup, False])

    test_trajectories = None

    return env, sc, interaction, train_trajectories, test_trajectories


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
        env, sc, interaction, train_trajectories, test_trajectories = load_cm_offline_data_from_csv(cfg)
    elif cfg.offline_data.load_from == 'transition':
        env, sc, interaction, train_trajectories, test_trajectories = load_cm_offline_data_from_transitions(cfg)

    # put all training transitions together in one list
    train_transitions = []
    for traj in train_trajectories:
        train_transitions += traj.transitions

    reward_func = init_reward_function(cfg.env.reward)
    train_info = {
        'train_trajectories': train_trajectories,
        'train_transitions': train_transitions,
        'reward_func': reward_func,
        'interaction': interaction,
        'test_trajectories': test_trajectories
    }
    print(train_transitions[0].obs)
    cm = init_calibration_model(cfg.calibration_model, train_info)
    cm.train()

    state_dim, action_dim = get_state_action_dim(env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)


if __name__ == "__main__":
    main()
