import hydra
import numpy as np
import torch
import random
import time

from omegaconf import DictConfig

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.interaction.factory import init_interaction
from corerl.utils.device import init_device
from corerl.data_loaders.factory import init_data_loader
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.data.transition_normalizer import TransitionNormalizer
from corerl.data_loaders.utils import train_test_split
from corerl.utils.plotting import make_plots

import corerl.utils.freezer as fr
import main_utils as utils


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    init_start = time.time()
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    test_epochs = cfg.experiment.test_epochs
    init_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    init_end = time.time()
    print("Path + Freezer + Seed Init Duration:", init_end - init_start)

    env_start = time.time()
    env = init_environment(cfg.env)
    env_end = time.time()

    do_offline_training = cfg.experiment.offline_steps > 0
    # first, load the data and potentially update the bounds of the obs space of the environment
    # it's important this happens before we create the state constructor and interaction since normalization
    # depends on these values
    if do_offline_training:
        dl_start = time.time()
        dl = init_data_loader(cfg.data_loader)
        dl_end = time.time()
        print("Data Loader Init Duration:", dl_end - dl_start)
        df_start = time.time()
        all_data_df, train_data_df, test_data_df = utils.load_df_from_csv(cfg, dl)
        df_end = time.time()
        print("DataFrame Loading Duration:", df_end - df_start)
        if cfg.experiment.load_env_obs_space_from_data:
            env_update_start = time.time()
            env = utils.set_env_obs_space(env, all_data_df, dl)
            env_update_end = time.time()
            print("Env Bounds Update Duration:", env_update_end - env_update_start)

    # the next part of instantiates objects. It is shared between online and offline training
    sc_start = time.time()
    sc = init_state_constructor(cfg.state_constructor, env)
    sc_end = time.time()
    print("State Constructor Init Duration:", sc_end - sc_start)
    state_dim, action_dim = utils.get_state_action_dim(env, sc)
    print("State Dim: {}, action dim: {}".format(state_dim, action_dim))
    agent_start = time.time()
    agent = init_agent(cfg.agent, state_dim, action_dim)
    agent_end = time.time()
    print("Agent Init Duration:", agent_end - agent_start)

    alert_args = {
        'agent': agent,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'input_dim': state_dim,
    }

    normalizer_start = time.time()
    obs_normalizer = ObsTransitionNormalizer(cfg.normalizer, env)
    transition_normalizer = TransitionNormalizer(cfg.normalizer, env)
    composite_alert = CompositeAlert(cfg.alerts, alert_args)
    transition_creator = AnytimeTransitionCreator(cfg.transition_creator, composite_alert)
    normalizer_end = time.time()
    print("Normalizers + Alerts + Transition Creator Init Duration:", normalizer_end - normalizer_start)

    plot_transitions = None
    agent_test_transitions = None

    if do_offline_training:
        print('Loading offline observations...')
        obs_transition_start = time.time()
        train_obs_transitions, test_obs_transitions = utils.get_offline_obs_transitions(cfg,
                                                                                        train_data_df,
                                                                                        test_data_df,
                                                                                        dl, obs_normalizer)
        obs_transition_end = time.time()
        print("Obs Transition Creation Duration:", obs_transition_end - obs_transition_start)

        print('Loading offline transitions...')
        state_transition_start = time.time()
        agent_train_transitions, agent_test_transitions, alert_train_transitions, alert_test_transitions, _ = utils.get_offline_transitions(
            cfg, train_obs_transitions,
            test_obs_transitions, sc,
            transition_creator)
        state_transition_end = time.time()
        print("State Transition Creation Duration:", state_transition_end - state_transition_start)

        split_start = time.time()
        all_agent_transitions = agent_train_transitions + agent_test_transitions
        dp_transitions = utils.get_dp_transitions(all_agent_transitions)
        split = train_test_split(dp_transitions, train_split=cfg.experiment.plot_split)
        plot_transitions = split[0][1]
        split_end = time.time()
        print("State Transition Split Duration:", split_end - split_start)

        agent_start = time.time()
        offline_eval = utils.offline_training(cfg,
                                              env,
                                              agent,
                                              agent_train_transitions,
                                              plot_transitions,
                                              save_path,
                                              test_epochs)
        agent_end = time.time()
        print("Offline Agent Training Duration:", agent_end - agent_start)

        alert_start = time.time()
        utils.offline_alert_training(cfg, composite_alert, alert_train_transitions)
        alert_end = time.time()
        print("Offline Alert Training Duration:", alert_end - alert_start)

    if not (test_epochs is None):
        assert not (plot_transitions is None), "Must include test transitions if test_epochs is not None"

    interaction_start = time.time()
    interaction = init_interaction(cfg.interaction, env, sc, composite_alert,
                                   transition_creator, obs_normalizer,
                                   transitions=agent_test_transitions)
    interaction_end = time.time()
    print("Interaction Init Duration:", interaction_end - interaction_start)

    if cfg.interaction.name == "offline_anytime":  # simulating online experience from an offline dataset
        offline_anytime_start = time.time()
        online_eval = utils.offline_anytime_deployment(cfg,
                                                       agent,
                                                       interaction,
                                                       env,
                                                       composite_alert,
                                                       transition_normalizer,
                                                       save_path,
                                                       plot_transitions,
                                                       test_epochs)
        offline_anytime_end = time.time()
        print("Offline Anytime Duration:", offline_anytime_end - offline_anytime_start)
        online_eval.output(save_path / 'stats.json')
    else:
        online_eval = utils.online_deployment(cfg,
                                              agent,
                                              interaction,
                                              env,
                                              composite_alert,
                                              transition_normalizer,
                                              save_path,
                                              plot_transitions,
                                              test_epochs)
        online_eval.output(save_path / 'stats.json')

    # need to update make_plots here
    stats = online_eval.get_stats()
    make_plots(fr.freezer, stats, save_path / 'plots')
    # env.plot(save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    # return stats


if __name__ == "__main__":
    main()
