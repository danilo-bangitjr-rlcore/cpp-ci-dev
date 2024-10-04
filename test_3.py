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
from corerl.data.factory import init_transition_creator
from corerl.data.transition_creator import OldAnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.data.transition_normalizer import TransitionNormalizer
from corerl.interaction.anytime_interaction import AnytimeInteraction, OldAnytimeInteraction
from tqdm import tqdm

import corerl.utils.freezer as fr
import main_utils as utils


def check_equal(t_1, t_2):
    assert len(t_1) == len(t_2), str(len(t_1)) + ', ' + str(len(t_2))
    for i, _ in enumerate(t_1):
        if not t_1[i] == t_2[i]:
            print(i)
            print("original")
            print(t_1[i])
            print("refactored")
            print(t_2[i])
            assert False


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

    env_1 = init_environment(cfg.env)
    env_2 = init_environment(cfg.env)

    sc = init_state_constructor(cfg.state_constructor, env_1)
    state_dim, action_dim = utils.get_state_action_dim(env_1, sc)
    print("State Dim: {}, action dim: {}".format(state_dim, action_dim))
    log.info("State Dim: {}, action dim: {}".format(state_dim, action_dim))
    agent = init_agent(cfg.agent, state_dim, action_dim)



    alert_args = {
        'agent': agent,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'input_dim': state_dim,
    }
    alerts = CompositeAlert(cfg.alerts, alert_args)
    alert_tc = init_transition_creator(cfg.alert_transition_creator, sc)
    alert_tc.init_alerts(alerts)

    obs_normalizer = ObsTransitionNormalizer(cfg.normalizer, env_1)
    agent_tc = init_transition_creator(cfg.agent_transition_creator, sc)
    interaction = AnytimeInteraction(cfg.interaction, env_1, sc, obs_normalizer, agent_tc)
    interaction.init_alerts(alerts, alert_tc)

    old_agent_tc = OldAnytimeTransitionCreator(cfg.old_agent_transition_creator, alerts)
    old_interaction = OldAnytimeInteraction(cfg.interaction, env_2, sc, alerts, old_agent_tc, obs_normalizer)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))

    state, info = interaction.reset()
    state_2, info = old_interaction.reset()

    assert np.all(np.isclose(state, state_2))
    action = agent.get_action(state)  # initial action

    for j in pbar:
        transitions, agent_train_transitions, _, alert_train_transitions, alert_info, env_info = interaction.step(action)

        for transition in agent_train_transitions:
            agent.update_buffer(transition)

        for transition in alert_train_transitions:
            alerts.update_buffer(transition)

        old_transitions, old_agent_train_transitions, old_alert_train_transitions, old_alert_info, old_env_info = old_interaction.step(action)

        # check_equal(old_transitions, transitions)
        check_equal(old_agent_train_transitions, agent_train_transitions)
        check_equal(old_alert_train_transitions, alert_train_transitions)

        agent.update()
        alerts.update()

        if len(transitions) > 0:
            state = transitions[-1].next_state
            action = agent.get_action(state)


if __name__ == "__main__":
    main()
