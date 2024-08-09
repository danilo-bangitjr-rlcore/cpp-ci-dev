import hydra
import numpy as np
import torch
import random
import time

from omegaconf import DictConfig

print("Import Device")
from corerl.utils.device import device
print("Import Init Agent")
from corerl.agent.factory import init_agent
print("Import Init Environment")
from corerl.environment.factory import init_environment
print("Import Init State Constructor")
from corerl.state_constructor.factory import init_state_constructor
print("Import Init Alert")
from corerl.alerts.composite_alert import CompositeAlert
print("Import Init Interaction")
from corerl.interaction.factory import init_interaction
print("Import Init Data Loader")
from corerl.data_loaders.factory import init_data_loader
print("Import Init Transition Creator")
from corerl.data.transition_creator import AnytimeTransitionCreator
print("Import Init ObsTransitionNormalizer")
from corerl.data.obs_normalizer import ObsTransitionNormalizer
print("Import Init TransitionNormalizer")
from corerl.data.transition_normalizer import TransitionNormalizer
print("Import Init Train Test Split")
from corerl.data_loaders.utils import train_test_split
print("Import Init Make Plots")
from corerl.utils.plotting import make_plots

import corerl.utils.freezer as fr
import main_utils as utils
print("End of main imports")

@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    print("Start Main")
    run_start = time.time()
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    test_epochs = cfg.experiment.test_epochs
    print("cfg.experiment.device:", cfg.experiment.device)
    device.update_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = init_environment(cfg.env)

    data_load_start = time.time()
    do_offline_training = cfg.experiment.offline_steps > 0
    # first, load the data and potentially update the bounds of the obs space of the environment
    # it's important this happens before we create the state constructor and interaction since normalization
    # depends on these values
    if do_offline_training:
        dl = init_data_loader(cfg.data_loader)
        all_data_df, train_data_df, test_data_df = utils.load_df_from_csv(cfg, dl)
        if cfg.experiment.load_env_obs_space_from_data:
            env = utils.set_env_obs_space(env, all_data_df, dl)

    # the next part of instantiates objects. It is shared between online and offline training
    sc = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = utils.get_state_action_dim(env, sc)
    print("State Dim: {}, action dim: {}".format(state_dim, action_dim))
    agent = init_agent(cfg.agent, state_dim, action_dim)

    alert_args = {
        'agent': agent,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'input_dim': state_dim,
    }

    obs_normalizer = ObsTransitionNormalizer(cfg.normalizer, env)
    transition_normalizer = TransitionNormalizer(cfg.normalizer, env)
    composite_alert = CompositeAlert(cfg.alerts, alert_args)
    transition_creator = AnytimeTransitionCreator(cfg.transition_creator, composite_alert)

    plot_transitions = None
    agent_test_transitions = None

    if do_offline_training:
        print('Loading offline observations...')
        train_obs_transitions, test_obs_transitions = utils.get_offline_obs_transitions(cfg,
                                                                                        train_data_df,
                                                                                        test_data_df,
                                                                                        dl, obs_normalizer)

        print('Loading offline transitions...')
        agent_train_transitions, agent_test_transitions, alert_train_transitions, alert_test_transitions, _ = utils.get_offline_transitions(
            cfg, train_obs_transitions,
            test_obs_transitions, sc,
            transition_creator)

        all_agent_transitions = agent_train_transitions + agent_test_transitions
        dp_transitions = utils.get_dp_transitions(all_agent_transitions)
        split = train_test_split(dp_transitions, train_split=cfg.experiment.plot_split)
        plot_transitions = split[0][1]
        data_load_end = time.time()
        print("Data Loading Time:", data_load_end - data_load_start)

        offline_start = time.time()
        offline_eval = utils.offline_training(cfg,
                                              env,
                                              agent,
                                              agent_train_transitions,
                                              plot_transitions,
                                              save_path,
                                              test_epochs)
        offline_end = time.time()
        print("Offline Training Time:", offline_end - offline_start)

        alert_start = time.time()
        utils.offline_alert_training(cfg, composite_alert, alert_train_transitions)
        alert_end = time.time()
        print("Offline Alert Training Time:", alert_end - alert_start)

    if not (test_epochs is None):
        assert not (plot_transitions is None), "Must include test transitions if test_epochs is not None"

    interaction = init_interaction(cfg.interaction, env, sc, composite_alert,
                                   transition_creator, obs_normalizer,
                                   transitions=agent_test_transitions)

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
        #online_eval.output(save_path / 'stats.json')
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
        #online_eval.output(save_path / 'stats.json')

    # need to update make_plots here
    stats = online_eval.get_stats()
    make_plots(fr.freezer, stats, save_path / 'plots')
    # env.plot(save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    run_end = time.time()
    print("Total Run Time:", run_end - run_start)

    # return stats


if __name__ == "__main__":
    main()
