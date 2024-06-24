import hydra
import numpy as np
import torch
import random

from omegaconf import DictConfig

from corerl.data_loaders.utils import make_anytime_trajectories
from corerl.calibration_models.factory import init_calibration_model
from corerl.data import Transition, Trajectory
from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.interaction.factory import init_interaction
from corerl.utils.device import init_device
import corerl.utils.freezer as fr
from main_utils import *


def trajectories_to_transitions(trajectories: list[Trajectory]) -> list[Transition]:
    transitions = []
    for t in trajectories:
        transitions += t.transitions
    return transitions


# def load_cm_offline_data_from_transitions(cfg: DictConfig) -> dict:
#     output_path = Path(cfg.offline_data.output_path)
#     env = init_environment(cfg.env)
#
#     # We assume that transitions have been created with make_offline_transitions.py
#     nothing_fn = lambda *args: None
#
#     train_obs_transitions = load_or_create(output_path,
#                                            [cfg.env, cfg.interaction, cfg.agent],
#                                            'obs_transitions', nothing_fn,
#                                            [])
#
#     sc_cm = init_state_constructor(cfg.calibration_model.state_constructor, env)
#
#     interaction_cm = init_interaction(cfg.interaction, env, sc_cm)
#     create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_trajectories(
#         obs_transitions,
#         interaction_,
#         sc_warmup=warmup,
#         steps_per_decision=cfg.interaction.steps_per_decision,
#         gamma=cfg.experiment.gamma,
#         return_scs=return_scs
#     )
#
#     # next, we will create the training and test transitions for the calibration model
#     train_trajectories_cm = load_or_create(output_path,
#                                            [cfg.data_loader, cfg.calibration_model.state_constructor, cfg.interaction],
#                                            'train_trajectories_cm', create_trajectories,
#                                            [train_obs_transitions, interaction_cm,
#                                             cfg.calibration_model.state_constructor.warmup, True])
#
#     # TODO: get this index from somewhere else
#     train_trajectories_cm, test_trajectories_cm = train_trajectories_cm[0].split_at(3999)
#     train_trajectories_cm = [train_trajectories_cm]
#     test_trajectories_cm = [test_trajectories_cm]
#
#     # finally, create the train and test transitions for the agent
#     sc_agent = init_state_constructor(cfg.state_constructor, env)
#     interaction_agent = init_interaction(cfg.interaction, env, sc_agent)
#
#     train_trajectories_agent = load_or_create(output_path,
#                                               [cfg.data_loader, cfg.state_constructor, cfg.interaction],
#                                               'train_trajectories_agent', create_trajectories,
#                                               [train_obs_transitions, interaction_agent, cfg.state_constructor.warmup,
#                                                True])
#
#     # TODO: get this index from somewhere else
#     train_trajectories_agent, test_trajectories_agent = train_trajectories_agent[0].split_at(3999)
#     train_trajectories_agent = [train_trajectories_agent]
#     test_trajectories_agent = [test_trajectories_agent]
#
#     return_dict = {
#         'env': env,
#         'sc_cm': sc_cm,
#         'agent_sc': sc_agent,
#         'interaction_cm': interaction_cm,
#         'interaction_agent': interaction_agent,
#         'train_trajectories_cm': train_trajectories_cm,
#         'train_trajectories_agent': train_trajectories_agent,
#         'test_trajectories_cm': test_trajectories_cm,
#         'test_trajectories_agent': test_trajectories_agent,
#         'train_transitions_cm': trajectories_to_transitions(train_trajectories_cm),
#         'train_transitions_agent': trajectories_to_transitions(train_trajectories_agent),
#         'test_transitions_cm': trajectories_to_transitions(test_trajectories_cm),
#         'test_transitions_agent': trajectories_to_transitions(test_trajectories_agent)
#     }
#
#     return return_dict
#

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

    env = init_environment(cfg.env)
    env, dl, train_obs_transitions, test_obs_transitions = load_offline_obs_from_csv(cfg, env)

    sc_agent = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = get_state_action_dim(env, sc_agent)
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

    # assume scs have the same warmup period for now
    assert cfg.state_constructor.warmup == cfg.calibration_model.state_constructor.warmup
    warmup = cfg.state_constructor.warmup

    # load trajectories for the agent
    interaction_agent = init_interaction(cfg.interaction, env, sc_agent, composite_alert, data_loader=dl)
    agent_hash_cfgs = [cfg.data_loader, cfg.state_constructor, cfg.interaction]
    train_trajectories_agent, _, test_trajectories_agent, _ = get_offline_trajectories(cfg,
                                                                                       agent_hash_cfgs,
                                                                                       train_obs_transitions,
                                                                                       test_obs_transitions,
                                                                                       interaction_agent,
                                                                                       composite_alert,
                                                                                       warmup=warmup,
                                                                                       )

    # load trajectories for the model
    print("loading trajectories for the model")
    sc_cm = init_state_constructor(cfg.calibration_model.state_constructor, env)
    cm_hash_cfgs = [cfg.data_loader, cfg.calibration_model.state_constructor, cfg.interaction]
    interaction_cm = init_interaction(cfg.interaction, env, sc_cm, composite_alert, data_loader=dl)
    train_trajectories_cm, _, test_trajectories_cm, _ = get_offline_trajectories(cfg,
                                                                                 cm_hash_cfgs,
                                                                                 train_obs_transitions,
                                                                                 test_obs_transitions,
                                                                                 interaction_cm,
                                                                                 composite_alert,
                                                                                 return_train_sc=True,
                                                                                 warmup=warmup,
                                                                                 )

    train_info = {
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

    reward_func = init_reward_function(cfg.env.reward)
    train_info["reward_func"] = reward_func
    cm = init_calibration_model(cfg.calibration_model, train_info)
    cm.train()

    # agent should be pretty bad here
    returns = cm.do_agent_rollouts(agent, train_info['test_trajectories_agent'], plot='pre_training',
                                   plot_save_path=save_path)
    cm.do_test_rollouts(save_path)
    print("returns", returns)
    print("mean return pre-training:", np.mean(returns))

    # now, train the agent, is it better?
    test_epochs = cfg.experiment.test_epochs
    offline_training(cfg,
                     env,
                     agent,
                     trajectories_to_transitions(train_trajectories_agent),
                     trajectories_to_transitions(test_trajectories_agent),
                     save_path,
                     test_epochs=test_epochs)

    # evaluate the agent
    returns = cm.do_agent_rollouts(agent, train_info['test_trajectories_agent'],
                                   plot='post_training',
                                   plot_save_path=save_path)
    print("returns", returns)
    print("mean return post-training:", np.mean(returns))


if __name__ == "__main__":
    main()
