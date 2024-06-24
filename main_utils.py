import numpy as np
import hashlib
import copy
import pickle as pkl

from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from gymnasium.spaces.utils import flatdim
from gymnasium import spaces, Env
from pathlib import Path
from typing import Optional

from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.eval.composite_eval import CompositeEval
from corerl.interaction.factory import init_interaction
from corerl.data_loaders.factory import init_data_loader
from corerl.data_loaders.base import BaseDataLoader
from corerl.environment.reward.factory import init_reward_function
from corerl.data_loaders.utils import make_anytime_transitions, train_test_split, make_anytime_trajectories
from corerl.agent.utils import get_test_state_qs_and_policy_params
from corerl.utils.plotting import visualize_actor_critic, plot_action_value_alert
from corerl.data import Transition, ObsTransition, Trajectory
from corerl.interaction.base import BaseInteraction
from corerl.alerts.composite_alert import CompositeAlert
from corerl.state_constructor.base import BaseStateConstructor
from corerl.agent.base import BaseAgent


def prepare_save_dir(cfg: DictConfig):
    if cfg.experiment.param_from_hash:
        cfg_copy = copy.deepcopy(cfg)
        del cfg_copy.experiment.seed
        cfg_hash = hashlib.sha1(str(cfg_copy).encode("utf-8")).hexdigest()
        print("Creating experiment param from hash:", cfg_hash)
        cfg.experiment.param = cfg_hash
    save_path = (
            Path(cfg.experiment.save_path) /
            cfg.experiment.exp_name /
            (f'param-{cfg.experiment.param}') /
            (f'seed-{cfg.experiment.seed}')
    )
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    return save_path


def update_pbar(pbar, stats: dict, keys: list) -> None:
    """
    Updates the pbar with the entries in stats for keys in keys
    """

    # keys = ['last_bellman_error', 'avg_reward']  # which information to display
    pbar_str = ''
    for k in keys:
        v = stats.get(k)
        if isinstance(v, float):
            pbar_str += '{key} : {val:.1f}, '.format(key=k, val=v)
        else:
            pbar_str += '{key} : {val} '.format(key=k, val=v)
    pbar.set_description(pbar_str)


def check_exists(save_path):
    if save_path.exists():
        with open(save_path, 'rb') as f:
            return pkl.load(f)
    else:
        return None


def merge_dictionaries(dict_1: dict, dict_2: dict) -> dict:
    """
    Merging Alert info dictionaries which are ultimately used for plotting
    """
    for key in dict_2:
        if type(dict_2[key]) == type([]):
            if key not in dict_1:
                dict_1[key] = dict_2[key]
            else:
                dict_1[key] += dict_2[key]
        elif type(dict_2[key]) == type({}):
            if key not in dict_1:
                dict_1[key] = dict_2[key]
            else:
                dict_1[key] = merge_dictionaries(dict_1[key], dict_2[key])

    return dict_1


def load_or_create(root: Path, cfgs: list[DictConfig], prefix: str, create_func: callable, args: list) -> object:
    """
    Will either load an object or create a new one using create func. Objects are saved at root using a hash determined
    by cfgs.
    """

    cfg_str = ''
    for cfg in cfgs:
        cfg_copy = OmegaConf.to_container(copy.deepcopy(cfg), resolve=True)
        cfg_str += str(cfg_copy)

    cfg_hash = hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()
    save_path = root / cfg_hash / f"{prefix}-{cfg_hash}.pkl"
    obj = check_exists(save_path)

    if obj is None:
        print(f"Generating {prefix}...")
        obj = create_func(*args)  # loads the entire dataset

        save_path = root / cfg_hash
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

        with open(save_path / f"{prefix}-{cfg_hash}.pkl", 'wb') as f:
            pkl.dump(obj, f)

        print(f"Saved {prefix} to {save_path}.")
    else:
        print(f"Loaded {prefix} from {save_path}.")

    return obj


def load_offline_obs_from_csv(cfg: DictConfig, env: Env) -> tuple[
    Env, BaseDataLoader, list[ObsTransition], Optional[list[ObsTransition]]]:
    """
    Loads offline observation transitions (transitions without states) from an offline dataset.

    As a side effect, will update an environments observation space to match the offline dataset
    """

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

    create_bounds = lambda dl_, df: dl_.get_obs_max_min(df)
    obs_bounds = load_or_create(output_path, [cfg.data_loader],
                                'obs_bounds', create_bounds, [dl, all_data_df])

    env.observation_space = spaces.Box(low=obs_bounds[0], high=obs_bounds[1], dtype=np.float32)
    print("Updated Env Observation Space:", env.observation_space)

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

    return env, dl, train_obs_transitions, test_obs_transitions


def get_offline_transitions(cfg: DictConfig,
                            train_obs_transitions: list[ObsTransition],
                            test_obs_transitions: list[ObsTransition],
                            interaction: BaseInteraction,
                            alerts: CompositeAlert) -> tuple[
    list[Transition], list[Transition], list[Transition], list[Transition], list[
        BaseStateConstructor]]:
    """
    Takes observation transitions and produces offline transitions (including state) using the interactions's state
    constructor=
    """
    output_path = Path(cfg.offline_data.output_path)

    create_transitions = lambda obs_transitions, interaction_, alerts_, warmup, return_scs: make_anytime_transitions(
        obs_transitions,
        interaction_,
        alerts_,
        sc_warmup=cfg.state_constructor.warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs)

    agent_train_transitions, alert_train_transitions, _ = load_or_create(output_path,
                                                                         [cfg.data_loader, cfg.state_constructor,
                                                                          cfg.interaction, cfg.alerts],
                                                                         'train_transitions', create_transitions,
                                                                         [train_obs_transitions, interaction, alerts,
                                                                          cfg.state_constructor.warmup, False])

    if test_obs_transitions is not None:
        agent_test_transitions, alert_test_transitions, test_scs = load_or_create(output_path,
                                                                                  [cfg.data_loader,
                                                                                   cfg.state_constructor,
                                                                                   cfg.interaction, cfg.alerts],
                                                                                  'test_transitions',
                                                                                  create_transitions,
                                                                                  [test_obs_transitions, interaction,
                                                                                   alerts, cfg.state_constructor.warmup,
                                                                                   True])
    else:
        agent_test_transitions = None
        alert_test_transitions = None
        test_scs = None

    return agent_train_transitions, alert_train_transitions, agent_test_transitions, alert_test_transitions, test_scs


def get_offline_trajectories(cfg: DictConfig,
                             hash_cfgs: list[DictConfig],
                             train_obs_transitions: list[ObsTransition],
                             test_obs_transitions: list[ObsTransition],
                             interaction: BaseInteraction,
                             alerts: CompositeAlert,
                             return_train_sc=False) -> tuple[
    list[Trajectory], list[Transition], list[Trajectory], list[Transition]]:
    """
    Takes observation transitions and produces offline transitions (including state) using the interaction's state
    constructor
    """
    output_path = Path(cfg.offline_data.output_path)
    create_trajectories = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_trajectories(
        obs_transitions,
        interaction_,
        alerts,
        sc_warmup=warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )

    alert_test_transitions = None
    # next, we will create the training and test transitions for the calibration model
    train_trajectories, alert_train_transitions, _ = load_or_create(output_path, hash_cfgs,
                                                                    'train_trajectories', create_trajectories,
                                                                    [train_obs_transitions, interaction,
                                                                     cfg.calibration_model.state_constructor.warmup,
                                                                     return_train_sc])
    if test_obs_transitions is not None:
        test_trajectories, alert_test_transitions, _ = load_or_create(output_path, hash_cfgs,
                                                                      'test_trajectories', create_trajectories,
                                                                      [test_obs_transitions, interaction,
                                                                       cfg.calibration_model.state_constructor.warmup,
                                                                       True])
    else:
        train_trajectories, test_trajectories = train_trajectories[0].split_at(3999)
        train_trajectories = [train_trajectories]
        test_trajectories = [test_trajectories]

    return train_trajectories, alert_train_transitions, test_trajectories, alert_test_transitions


def load_offline_data_from_transitions(cfg):  # TODO: test this out when I test saturation
    output_path = Path(cfg.offline_data.output_path)

    # We assume that transitions have been created with make_offline_transitions.py
    nothing_fn = lambda *args: None

    train_obs_transitions = load_or_create(output_path,
                                           [cfg.env, cfg.state_constructor, cfg.interaction, cfg.agent],
                                           'obs_transitions', nothing_fn,
                                           [])

    test_obs_transitions = None

    env = init_environment(cfg.env)
    sc = init_state_constructor(cfg.state_constructor, env)
    interaction = init_interaction(cfg.interaction, env, sc)

    create_transitions = lambda obs_transitions, interaction_, warmup, return_scs: make_anytime_transitions(
        obs_transitions,
        interaction_,
        sc_warmup=cfg.state_constructor.warmup,
        steps_per_decision=cfg.interaction.steps_per_decision,
        gamma=cfg.experiment.gamma,
        return_scs=return_scs
    )
    train_transitions, _ = load_or_create(output_path,
                                          [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                          'train_transitions', create_transitions,
                                          [train_obs_transitions, interaction, cfg.state_constructor.warmup, False])

    if test_obs_transitions is not None:
        test_transitions, test_scs = load_or_create(output_path,
                                                    [cfg.data_loader, cfg.state_constructor, cfg.interaction],
                                                    'test_transitions', create_transitions,
                                                    [test_obs_transitions, interaction, cfg.state_constructor.warmup,
                                                     True])
    else:
        test_transitions = None
        test_scs = None

    return env, sc, interaction, train_transitions, test_transitions, test_scs


def get_state_action_dim(env: Env, sc: BaseStateConstructor) -> tuple[int, int]:
    obs_shape = (flatdim(env.observation_space),)
    dummy_obs = np.ones(obs_shape)
    action_shape = (flatdim(env.action_space),)
    dummy_action = np.ones(action_shape)
    state_dim = sc.get_state_dim(dummy_obs, dummy_action)  # gets state_dim dynamically
    action_dim = flatdim(env.action_space)
    return state_dim, action_dim


def offline_alert_training(cfg: DictConfig, alerts: CompositeAlert, alert_train_transitions: list[Transition]) -> None:
    print('Starting offline alert training...')
    print("Num alert train transitions:", len(alert_train_transitions))
    for transition_tup in alert_train_transitions:
        alerts.update_buffer(transition_tup)

    offline_steps = cfg.experiment.offline_steps
    pbar = tqdm(range(offline_steps))
    for i in pbar:
        alerts.update()


def offline_training(cfg: DictConfig,
                     env: Env,
                     agent: BaseAgent,
                     agent_train_transitions: list[Transition],
                     agent_test_transitions: list[Transition],
                     save_path: Path,
                     test_epochs: Optional[list[int]] = None) -> CompositeEval:
    if test_epochs is None:
        test_epochs = []

    print('Starting offline agent training...')
    offline_eval_args = {
        'agent': agent
    }
    offline_eval = CompositeEval(cfg.eval, offline_eval_args, offline=True)

    if agent_test_transitions is None:
        split = train_test_split(agent_train_transitions, train_split=cfg.experiment.train_split)
        agent_train_transitions, agent_test_transitions = split[0][0], split[0][1]

    print("Num agent train transitions:", len(agent_train_transitions))
    for transition in agent_train_transitions:
        agent.update_buffer(transition)

    offline_steps = cfg.experiment.offline_steps
    pbar = tqdm(range(offline_steps))
    keys = cfg.experiment.offline_stat_keys  # which keys to log on the progress bar
    for i in pbar:
        agent.update()
        offline_eval.do_eval(**offline_eval_args)  # run all evaluators
        stats = offline_eval.get_stats()

        if i in test_epochs:
            test_states, test_actions, test_q_values, actor_params = get_test_state_qs_and_policy_params(agent,
                                                                                                         agent_test_transitions)
            visualize_actor_critic(test_states, test_actions, test_q_values, actor_params, env, save_path,
                                   "Offline_Training", i)

        update_pbar(pbar, stats, keys)

    return offline_eval


def online_deployment(cfg: DictConfig,
                      agent: BaseAgent,
                      interaction: BaseInteraction,
                      env: Env,
                      alerts: CompositeAlert,
                      save_path: Path,
                      agent_test_transitions: Optional[list[Transition]] = None,
                      test_epochs: Optional[list[Transition]] = None):
    if test_epochs is None:
        test_epochs = []

    # Instantiate online evaluators
    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    alerts_plot_info = {}
    state, info = interaction.reset()
    # State Warmup Here?
    print('Starting online training...')
    for j in pbar:
        action = agent.get_action(state)
        new_agent_transitions, agent_train_transitions, alert_train_transitions, alert_info_list, env_info_list = interaction.step(
            action)

        for transition in agent_train_transitions:
            agent.update_buffer(transition)

        for transition_tup in alert_train_transitions:
            alerts.update_buffer(transition_tup)

        agent.update()
        alerts.update()

        # logging + evaluation
        # union of the information needed by all evaluators
        online_eval_args = {
            'agent': agent,
            'env': env,
            'transitions': new_agent_transitions
        }

        online_eval.do_eval(**online_eval_args)
        stats = online_eval.get_stats()
        update_pbar(pbar, stats)

        for i in range(len(alert_info_list)):
            alerts_plot_info = merge_dictionaries(alerts_plot_info, alert_info_list[i])

        if j in test_epochs:
            test_states, test_actions, test_q_values, actor_params = get_test_state_qs_and_policy_params(agent,
                                                                                                         agent_test_transitions)
            visualize_actor_critic(test_states, test_actions, test_q_values, actor_params, env, save_path,
                                   "Online_Deployment", j)

        terminated = new_agent_transitions[-1].terminated
        truncated = new_agent_transitions[-1].truncate
        if terminated or truncated:
            state, _ = interaction.reset()
        else:
            state = new_agent_transitions[-1].next_state

    # Make Alerts Plot
    alert_trace_thresholds = alerts.get_trace_thresh()
    plot_action_value_alert(alerts_plot_info, alert_trace_thresholds, save_path)

    return online_eval


def offline_anytime_deployment(cfg: DictConfig,
                               agent: BaseAgent,
                               interaction: BaseInteraction,
                               env: Env,
                               alerts: CompositeAlert,
                               save_path: Path,
                               agent_test_transitions: Optional[list[Transition]] = None,
                               test_epochs: Optional[list[int]] = None) -> CompositeEval:
    # Online Deployment
    # Instantiate online evaluators
    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    interaction.warmup_sc()
    alerts_plot_info = {}
    pbar = tqdm(range(max_steps))
    print('Starting online anytime training with offline dataset...')
    for j in pbar:
        agent_transitions, agent_train_transitions, alert_train_transitions, info_list = interaction.step()

        for transition in agent_train_transitions:
            agent.update_buffer(transition)

        for transition_tup in alert_train_transitions:
            alerts.update_buffer(transition_tup)

        agent.update()
        alerts.update()

        # logging + evaluation
        # union of the information needed by all evaluators
        online_eval_args = {
            'agent': agent,
            'env': env,
            'transitions': agent_transitions
        }

        online_eval.do_eval(**online_eval_args)
        stats = online_eval.get_stats()

        if j in test_epochs:
            test_states, test_actions, test_q_values, actor_params = get_test_state_qs_and_policy_params(agent,
                                                                                                         agent_test_transitions)
            visualize_actor_critic(test_states, test_actions, test_q_values, actor_params, env, save_path,
                                   "Online_Interaction_Offline_Data", j)

        for i in range(len(info_list)):
            alerts_plot_info = merge_dictionaries(alerts_plot_info, info_list[i])

        update_pbar(pbar, stats)

        if agent_transitions[-1].truncate:
            print("Reached End Of Offline Eval Data")
            break

    # Make Alerts Plot
    alert_trace_thresholds = alerts.get_trace_thresh()
    plot_action_value_alert(alerts_plot_info, alert_trace_thresholds, save_path)

    return online_eval
