import logging
from omegaconf import DictConfig
import hydra
import numpy as np
import torch
import random

log = logging.getLogger(__name__)

from corerl.utils.device import device
from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data_loaders.factory import init_data_loader_test
from corerl.data.transition_creator import OldAnytimeTransitionCreator, AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.data.transition_normalizer import TransitionNormalizer

import corerl.utils.freezer as fr
import main_utils as utils


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    test_epochs = cfg.experiment.test_epochs
    device.update_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = init_environment(cfg.env)

    do_offline_training = cfg.experiment.offline_steps > 0
    # first, load the data and potentially update the bounds of the obs space of the environment
    # it's important this happens before we create the state constructor and interaction since normalization
    # depends on these values

    dl, dl_refac = init_data_loader_test(cfg.data_loader)

    all_data_df, train_data_df, test_data_df = utils.load_df_from_csv(cfg, dl)
    all_data_df_r, train_data_df_r, test_data_df_r = utils.load_df_from_csv(cfg, dl_refac)

    assert all_data_df.equals(all_data_df_r)

    if cfg.experiment.load_env_obs_space_from_data:
        env = utils.set_env_obs_space(env, all_data_df, dl)

    sc = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = utils.get_state_action_dim(env, sc)
    print("State Dim: {}, action dim: {}".format(state_dim, action_dim))
    log.info("State Dim: {}, action dim: {}".format(state_dim, action_dim))
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
    old_transition_creator = OldAnytimeTransitionCreator(cfg.transition_creator, composite_alert)

    transition_creator = AnytimeTransitionCreator(cfg.transition_creator, sc)
    transition_creator.init_alerts(composite_alert)

    plot_transitions = None
    agent_test_transitions = None

    # THIS IS THE OLD WAY
    print('Loading offline observations...')
    log.info('Loading offline observations...')
    train_obs_transitions, test_obs_transitions = utils.get_offline_obs_transitions(cfg,
                                                                                    train_data_df,
                                                                                    test_data_df,
                                                                                    dl,
                                                                                    obs_normalizer)

    train_obs_transitions_r, test_obs_transitions_r = utils.get_offline_obs_transitions(cfg,
                                                                                        train_data_df,
                                                                                        test_data_df,
                                                                                        dl_refac,
                                                                                        obs_normalizer,
                                                                                        prefix='refac_'
                                                                                        )

    # train_obs_transitions, test_obs_transitions = train_obs_transitions[:1000], test_obs_transitions[:60]
    # train_obs_transitions_r, test_obs_transitions_r = train_obs_transitions_r[:1000], test_obs_transitions_r[:60]

    def check_equal(obs_t_1, obs_t_2):
        assert len(obs_t_1) == len(obs_t_2)
        for i, _ in enumerate(obs_t_1):

            o_1 = obs_t_1[i]
            o_2 = obs_t_2[i]

            if not (np.allclose(o_1.obs, o_2.obs)
                    and np.allclose(o_1.action, o_2.action)
                    and np.allclose(o_1.next_obs, o_2.next_obs)
                    and o_1.reward == o_2.reward):
                print(i, "/", len(obs_t_1))
                print('original')
                print(o_1)
                print('refactored')
                print(o_2)
                assert False


        print("passed obs transition test :) ")

    def check_equal2(t_1, t_2):
        # assert len(t_1) == len(t_2), str(len(t_1)) + ', ' + str(len(t_2))
        for i, _ in enumerate(t_1):
            if not t_1[i] == t_2[i]:
                print(i)
                print("original")
                print(t_1[i])
                print("refactored")
                print(t_2[i])
                assert False

        print("passed transition test :) ")

    check_equal(train_obs_transitions, train_obs_transitions_r)
    check_equal(test_obs_transitions, test_obs_transitions_r)

    print('Loading offline transitions...')
    log.info('Loading offline transitions...')
    agent_train_transitions, agent_test_transitions, alert_train_transitions, alert_test_transitions = utils.get_offline_transitions(
        cfg,
        train_obs_transitions,
        test_obs_transitions,
        sc,
        old_transition_creator)

    # needs a new transition_creator

    agent_train_transitions_refac, agent_test_transitions_refac = utils.get_offline_transitions_refactored(
        cfg,
        train_obs_transitions_r,
        test_obs_transitions_r,
        sc,
        transition_creator,
        prefix='refac_')

    check_equal2(agent_train_transitions, agent_train_transitions_refac)
    check_equal2(agent_test_transitions, agent_test_transitions_refac)


if __name__ == "__main__":
    main()
