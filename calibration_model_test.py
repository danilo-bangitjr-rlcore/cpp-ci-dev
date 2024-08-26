import hydra
import numpy as np
import torch
import random
import pickle

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from corerl.calibration_models.factory import init_calibration_model
from corerl.data.data import Transition, Trajectory
from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.data_loaders.factory import init_data_loader
from corerl.utils.device import init_device
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.environment.reward.factory import init_reward_function
from corerl.utils.plotting import make_actor_critic_plots
from corerl.eval.composite_eval import CompositeEval
from corerl.data_loaders.utils import train_test_split

import corerl.utils.freezer as fr
import main_utils as utils


def trajectories_to_transitions(trajectories: list[Trajectory]) -> list[Transition]:
    transitions = []
    for t in trajectories:
        transitions += t.transitions
    return transitions


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: DictConfig) -> dict:
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')

    init_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = init_environment(cfg.env)

    dl = init_data_loader(cfg.data_loader)
    all_data_df, train_data_df, test_data_df = utils.load_df_from_csv(cfg, dl)
    if cfg.experiment.load_env_obs_space_from_data:
        env = utils.set_env_obs_space(env, all_data_df, dl)

    sc_agent = init_state_constructor(cfg.state_constructor, env)
    state_dim, action_dim = utils.get_state_action_dim(env, sc_agent)
    print("State Dim:", state_dim)
    print("Action Dim:", action_dim)
    agent = init_agent(cfg.agent, state_dim, action_dim)

    alert_args = {
        'agent': agent,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'input_dim': state_dim,
    }
    normalizer = ObsTransitionNormalizer(cfg.normalizer, env)
    composite_alert = CompositeAlert(cfg.alerts, alert_args)
    transition_creator = AnytimeTransitionCreator(cfg.transition_creator, composite_alert)
    train_obs_transitions, test_obs_transitions = utils.get_offline_obs_transitions(cfg,
                                                                                    train_data_df,
                                                                                    test_data_df,
                                                                                    dl, normalizer)

    agent_hash_cfgs = [cfg.data_loader, cfg.state_constructor, cfg.interaction]
    warmup = cfg.state_constructor.warmup

    # load trajectories for the model
    print("loading trajectories for the model")
    sc_cm = init_state_constructor(cfg.calibration_model.state_constructor, env)

    # the models need all transitions, not just DP transitions
    transition_creator.set_only_dp_transitions(False)
    OmegaConf.update(cfg, "interaction.only_dp_transitions", False)
    cm_hash_cfgs = [cfg.data_loader, cfg.calibration_model.state_constructor, cfg.interaction]

    # load the transition
    train_trajectories_cm, test_trajectories_cm = utils.get_offline_trajectories(cfg,
                                                                                 cm_hash_cfgs,
                                                                                 train_obs_transitions,
                                                                                 test_obs_transitions,
                                                                                 sc_cm,
                                                                                 transition_creator,
                                                                                 warmup,
                                                                                 prefix='model')

    train_info = {
        'normalizer': normalizer,
        'train_trajectories_cm': train_trajectories_cm,
        'test_trajectories_cm': test_trajectories_cm,
        'train_transitions_cm': trajectories_to_transitions(train_trajectories_cm),
        'test_transitions_cm': trajectories_to_transitions(test_trajectories_cm),
        'transition_creator': transition_creator
    }

    # reset the transition creator to use whether the interactions settings are for dp transitions
    transition_creator.set_only_dp_transitions(cfg.interaction.only_dp_transitions)

    reward_func = init_reward_function(cfg.env.reward)
    train_info["reward_func"] = reward_func

    cm = init_calibration_model(cfg.calibration_model, train_info)
    cm.train()

    print("Doing test rollouts...")
    losses = cm.do_test_rollouts(save_path / 'test_rollouts')

    with open(save_path / 'test_rollout_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)


if __name__ == "__main__":
    main()
