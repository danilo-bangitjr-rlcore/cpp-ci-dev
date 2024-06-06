from main import *

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
        'test_scs' : test_scs
    }

    # need to update make_plots here
    stats = online_eval.get_stats()
    make_plots(fr.freezer, stats, save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    return stats


if __name__ == "__main__":
    main()
