import hydra
import numpy as np
import torch
import random

from omegaconf import DictConfig

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.interaction.factory import init_interaction
from corerl.utils.device import init_device
import corerl.utils.freezer as fr
import main_utils as utils


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    test_epochs = cfg.experiment.test_epochs

    init_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = init_environment(cfg.env)
    dl = None
    train_obs_transitions = None
    test_obs_transitions = None
    agent_test_transitions = None

    do_offline_training = cfg.experiment.offline_steps > 0
    if do_offline_training:
        print('Loading offline observations...')
        # pass in env because load_offline_obs_from_csv updates env's observation space
        env, dl, train_obs_transitions, test_obs_transitions = utils.load_offline_obs_from_csv(cfg, env)

    sc = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = utils.get_state_action_dim(env, sc)
    print("State Dim:", state_dim)
    print("Action Dim:", action_dim)
    agent = init_agent(cfg.agent, state_dim, action_dim)

    alert_args = {
        'agent': agent,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'input_dim': state_dim,
    }
    composite_alert = CompositeAlert(cfg.alerts, alert_args)

    # the dataloader is optionally needed for interactions that read from the dataset. It is defaulted to None
    # if we do not load any offline data
    interaction = init_interaction(cfg.interaction, env, sc, composite_alert, data_loader=dl)

    if do_offline_training:
        print('Loading offline transitions...')
        agent_train_transitions, alert_train_transitions, agent_test_transitions, alert_test_transitions, _ = utils.get_offline_transitions(cfg,
                                                                                                                                            train_obs_transitions,
                                                                                                                                            test_obs_transitions,
                                                                                                                                            interaction,
                                                                                                                                            composite_alert)

        utils.offline_alert_training(cfg, composite_alert, alert_train_transitions)
        offline_eval = utils.offline_training(cfg,
                                              env,
                                              agent,
                                              agent_train_transitions,
                                              agent_test_transitions,
                                              save_path,
                                              test_epochs)

    if not (test_epochs is None):
        assert not (agent_test_transitions is None), "Must include test transitions if test_epochs is not None"

    if cfg.interaction.name == "offline_anytime":  # simulating online experience from an offline dataset
        online_eval = utils.offline_anytime_deployment(cfg,
                                                       agent,
                                                       interaction,
                                                       env,
                                                       composite_alert,
                                                       save_path,
                                                       agent_test_transitions,
                                                       test_epochs)
        online_eval.output(save_path / 'stats.json')
    else:
        online_eval = utils.online_deployment(cfg,
                                              agent,
                                              interaction,
                                              env, composite_alert,
                                              save_path,
                                              agent_test_transitions,
                                              test_epochs)
        online_eval.output(save_path / 'stats.json')


    # env.plot()
    # need to update make_plots here
    stats = online_eval.get_stats()
    # make_plots(fr.freezer, stats, save_path / 'plots')

    agent.save(save_path / 'agent')
    agent.load(save_path / 'agent')

    return stats


if __name__ == "__main__":
    main()
