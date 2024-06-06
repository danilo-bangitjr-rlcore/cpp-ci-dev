from main import *
from corerl.data_loaders.utils import make_anytime_trajectories

def load_offline_data(cfg):
    env = init_environment(cfg.env)
    dl = init_data_loader(cfg.data_loader)

    output_path = Path(cfg.offline_data_output_path)

    create_df = lambda dl_, filenames: dl_.load_data(filenames)
    all_data_df = load_or_create(output_path, [cfg.data_loader],
                                 'all_data_df', create_df, [dl, dl.all_filenames])
    train_data_df = load_or_create(output_path, [cfg.data_loader],
                                   'train_data_df', create_df, [dl, dl.train_filenames])
    test_data_df = load_or_create(output_path, [cfg.data_loader],
                                  'test_data_df', create_df, [dl, dl.test_filenames])

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

    create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_trajectories(
        obs_transitions,
        interaction_,
        sc_warmup=cfg.state_constructor.warmup,
        return_scs=return_scs)

    train_transitions, _ = load_or_create(output_path,
                                          [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                          'train_transitions', create_trajectories,
                                          [train_obs_transitions, interaction, cfg.state_constructor.warmup, False])

    if test_obs_transitions is not None:
        test_transitions, test_scs = load_or_create(output_path,
                                                    [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                                    'test_transitions', create_trajectories,
                                                    [test_obs_transitions, interaction, cfg.state_constructor.warmup,
                                                     True])
    else:
        test_transitions = None
        test_scs = None

    print("Done loading data!")
    return env, sc, interaction, train_transitions, test_transitions, test_scs


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


    env, sc, interaction, train_transitions, test_transitions, test_scs = load_offline_data(cfg)
    state_dim, action_dim = get_state_action_dim(env, sc)
    agent = init_agent(cfg.agent, state_dim, action_dim)
    reward_func = init_reward_function(cfg.env.reward)


    online_eval = online_deployment(cfg, agent, interaction, env)
    online_eval.output(save_path / 'stats.json')

    calibration_model_args = {
        'train_transitions': train_transitions,
        'reward_func': reward_func,
        'interaction' : interaction,
        'test_scs' : test_scs,
        # TODO: trajectories
    }

    # need to update make_plots here
    stats = online_eval.get_stats()
    make_plots(fr.freezer, stats, save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    return stats


if __name__ == "__main__":
    main()
