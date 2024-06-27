import numpy as np
import hashlib
import copy
import pickle as pkl

import pandas as pd
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
from corerl.data_loaders.base import BaseDataLoader
from corerl.environment.reward.factory import init_reward_function
from corerl.data_loaders.utils import train_test_split
from corerl.agent.utils import get_test_state_qs_and_policy_params
from corerl.utils.plotting import visualize_actor_critic, plot_action_value_alert
from corerl.data.data import Transition, ObsTransition, Trajectory
from corerl.interaction.base import BaseInteraction
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.transition_creator import AnytimeTransitionCreator
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


def set_env_obs_space(env: Env, df: pd.DataFrame, dl: BaseDataLoader):
    obs_bounds = dl.get_obs_max_min(df)
    env.observation_space = spaces.Box(low=obs_bounds[0], high=obs_bounds[1], dtype=np.float32)
    print("Updated env observation space:", env.observation_space)
    return env


def load_df_from_csv(cfg: DictConfig, dl: BaseDataLoader) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_path = Path(cfg.offline_data.output_path)

    def _create_df(filenames):
        return dl.load_data(filenames)

    all_data_df = load_or_create(root=output_path,
                                 cfgs=[cfg.data_loader],
                                 prefix='all_data_df',
                                 create_func=_create_df,
                                 args=[dl, dl.all_filenames])

    train_data_df = load_or_create(root=output_path,
                                   cfgs=[cfg.data_loader],
                                   prefix='train_data_df',
                                   create_func=_create_df,
                                   args=[dl.train_filenames])

    test_data_df = load_or_create(root=output_path,
                                  cfgs=[cfg.data_loader],
                                  prefix='test_data_df',
                                  create_func=_create_df,
                                  args=[dl.test_filenames])

    assert not all_data_df.isnull().values.any()
    assert not train_data_df.isnull().values.any()
    assert not test_data_df.isnull().values.any()
    assert not np.isnan(all_data_df.to_numpy()).any()

    return all_data_df, train_data_df, test_data_df


def get_offline_obs_transitions(cfg: DictConfig,
                                train_data_df: pd.DataFrame,
                                test_data_df: pd.DataFrame,
                                dl: BaseDataLoader,
                                normalizer: ObsTransitionNormalizer) -> tuple[
    list[ObsTransition], Optional[list[ObsTransition]]]:
    """
    Loads offline observation transitions (transitions without states) from an offline dataset.
    """

    output_path = Path(cfg.offline_data.output_path)
    reward_func = init_reward_function(cfg.env.reward)

    def _create_obs_transitions(df):
        return dl.create_obs_transitions(df, normalizer, reward_func)

    train_obs_transitions = load_or_create(root=output_path,
                                           cfgs=[cfg.data_loader],
                                           prefix='train_obs_transitions',
                                           create_func=_create_obs_transitions,
                                           args=[train_data_df])

    if test_data_df is not None:
        test_obs_transitions = load_or_create(root=output_path,
                                              cfgs=[cfg.data_loader],
                                              prefix='test_obs_transitions',
                                              create_func=_create_obs_transitions,
                                              args=[test_data_df])
    else:
        test_obs_transitions = None

    print(f"Loaded {len(train_obs_transitions)} train and {len(test_obs_transitions)} test obs transitions. ")

    return train_obs_transitions, test_obs_transitions


def get_offline_transitions(cfg: DictConfig,
                            train_obs_transitions: list[ObsTransition],
                            test_obs_transitions: list[ObsTransition],
                            sc: BaseStateConstructor,
                            transition_creator: AnytimeTransitionCreator,
                            return_train_scs: bool = False,
                            return_test_scs: bool = False) -> tuple[
    list[Transition], list[Transition], list[BaseStateConstructor]]:
    """
    Takes observation transitions and produces offline transitions (including state) using the interactions's state
    constructor=
    """
    output_path = Path(cfg.offline_data.output_path)

    def create_transitions(obs_transitions, return_scs):
        return transition_creator.make_offline_transitions(obs_transitions, sc, return_scs, use_pbar=True)

    train_transitions, _ = load_or_create(root=output_path,
                                          cfgs=[cfg.data_loader, cfg.state_constructor, cfg.transition_creator,
                                                cfg.alerts],
                                          prefix='train_transitions',
                                          create_func=create_transitions,
                                          args=[train_obs_transitions, return_train_scs])

    if test_obs_transitions is not None:
        test_transitions, test_scs = load_or_create(root=output_path,
                                                    cfgs=[cfg.data_loader, cfg.state_constructor,
                                                          cfg.transition_creator,
                                                          cfg.alerts],
                                                    prefix='test_transitions',
                                                    create_func=create_transitions,
                                                    args=[test_obs_transitions, return_test_scs])

    else:
        test_transitions = None
        test_scs = None

    print(f"Loaded {len(train_transitions)} train and {len(test_transitions)} test transitions. ")
    return train_transitions, test_transitions, test_scs


def get_offline_trajectories(cfg: DictConfig,
                             hash_cfgs: list[DictConfig],
                             train_obs_transitions: list[ObsTransition],
                             test_obs_transitions: list[ObsTransition],
                             sc: BaseStateConstructor,
                             transition_creator: AnytimeTransitionCreator,
                             return_train_scs: bool = False,
                             return_test_scs: bool = False,
                             warmup=0) -> tuple[list[Trajectory, list[Trajectory]]]:
    """
    Takes observation transitions and produces offline trajectories (including state) using the interactions's state
    constructor=
    """
    output_path = Path(cfg.offline_data.output_path)

    def create_trajectories(obs_transitions, return_scs):
        return transition_creator.make_offline_trajectories(obs_transitions, sc, return_scs,
                                                            use_pbar=True, warmup=warmup)

    train_trajectories = load_or_create(root=output_path,
                                        cfgs=hash_cfgs,
                                        prefix='train_trajectories',
                                        create_func=create_trajectories,
                                        args=[train_obs_transitions, return_train_scs])

    if test_obs_transitions is not None:
        test_trajectories = load_or_create(root=output_path,
                                           cfgs=hash_cfgs,
                                           prefix='test_transitions',
                                           create_func=create_trajectories,
                                           args=[test_obs_transitions, return_test_scs])

    else:
        test_transitions = []

    print(f"Loaded {len(train_trajectories)} train and {len(test_trajectories)} test trajectories. ")
    return train_trajectories, test_trajectories


def load_offline_data_from_transitions(cfg):  # TODO: update
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


def offline_alert_training(cfg: DictConfig, alerts: CompositeAlert, train_transitions: list[Transition]) -> None:
    print('Starting offline alert training...')
    print("Num alert train transitions:", len(train_transitions))
    # for transition_tup in alert_train_transitions:
    #     alerts.update_buffer(transition_tup)
    # TODO: verify with Alex this is correct

    alerts.update_buffer(train_transitions)
    offline_steps = cfg.experiment.offline_steps
    pbar = tqdm(range(offline_steps))
    for _ in pbar:
        alerts.update()


def offline_training(cfg: DictConfig,
                     env: Env,
                     agent: BaseAgent,
                     train_transitions: list[Transition],
                     test_transitions: list[Transition],
                     save_path: Path,
                     test_epochs: Optional[list[int]] = None) -> CompositeEval:
    if test_epochs is None:
        test_epochs = []

    print('Starting offline agent training...')
    offline_eval_args = {
        'agent': agent
    }
    offline_eval = CompositeEval(cfg.eval, offline_eval_args, offline=True)

    if test_transitions is None:
        split = train_test_split(train_transitions, train_split=cfg.experiment.train_split)
        train_transitions, test_transitions = split[0][0], split[0][1]

    print("Num agent train transitions:", len(train_transitions))
    for transition in train_transitions:
        agent.update_buffer(transition)

    offline_steps = cfg.experiment.offline_steps
    pbar = tqdm(range(offline_steps))
    for i in pbar:
        agent.update()
        offline_eval.do_eval(**offline_eval_args)  # run all evaluators
        stats = offline_eval.get_stats()

        if i in test_epochs:
            test_states, test_actions, test_q_values, actor_params = get_test_state_qs_and_policy_params(agent,
                                                                                                         test_transitions)
            visualize_actor_critic(test_states, test_actions, test_q_values, actor_params, env, save_path,
                                   "Offline_Training", i)

        update_pbar(pbar, stats, cfg.experiment.offline_stat_keys)

    return offline_eval


def online_deployment(cfg: DictConfig,
                      agent: BaseAgent,
                      interaction: BaseInteraction,
                      env: Env,
                      alerts: CompositeAlert,
                      save_path: Path,
                      plot_transitions: Optional[list[Transition]] = None,
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
    action = agent.get_action(state)  # initial action
    # State Warmup Here?
    print('Starting online training...')
    for j in pbar:
        transitions, train_transitions, alert_info_list, env_info_list = interaction.step(action)

        for transition in transitions:  # agent_train_transitions:
            agent.update_buffer(transition)

        if agent.buffer.size > 0:
            agent.update()

        if len(transitions) > 0:
            # for transition_tup in train_transitions:
            #     alerts.update_buffer(transition_tup)

            # alerts.update() # TODO: add this back

            # logging + evaluation
            # union of the information needed by all evaluators
            online_eval_args = {
                'agent': agent,
                'env': env,
                'transitions': transitions
            }

            online_eval.do_eval(**online_eval_args)
            stats = online_eval.get_stats()
            update_pbar(pbar, stats, cfg.experiment.online_stat_keys)

            # for i in range(len(alert_info_list)):
            #     alerts_plot_info = merge_dictionaries(alerts_plot_info, alert_info_list[i])

            terminated = transitions[-1].terminated
            truncated = transitions[-1].truncate
            if terminated or truncated:
                state, _ = interaction.reset()
            else:
                state = transitions[-1].next_state

            action = agent.get_action(state)

        if j in test_epochs:
            test_states, test_actions, test_q_values, actor_params = get_test_state_qs_and_policy_params(agent,
                                                                                                         plot_transitions)
            visualize_actor_critic(test_states, test_actions, test_q_values, actor_params, env, save_path,
                                   "Online_Deployment", j)

    # Make Alerts Plot
    # alert_trace_thresholds = alerts.get_trace_thresh()
    # plot_action_value_alert(alerts_plot_info, alert_trace_thresholds, save_path)

    return online_eval


def offline_anytime_deployment(cfg: DictConfig,
                               agent: BaseAgent,
                               interaction: BaseInteraction,
                               env: Env,
                               alerts: CompositeAlert,
                               save_path: Path,
                               plot_transitions: Optional[list[Transition]] = None,
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
        transitions, train_transitions, alert_info, _ = interaction.step()  # does not need an action from the agent

        if transitions is None:
            print("Reached End Of Offline Eval Data")
            break

        for transition in train_transitions:
            agent.update_buffer(transition)
            alerts.update_buffer(transition)

        agent.update()
        alerts.update()

        # logging + evaluation
        # union of the information needed by all evaluators
        online_eval_args = {
            'agent': agent,
            'env': env,
            'transitions': transitions
        }

        online_eval.do_eval(**online_eval_args)
        stats = online_eval.get_stats()

        if j in test_epochs:
            test_states, test_actions, test_q_values, actor_params = get_test_state_qs_and_policy_params(agent,
                                                                                                         plot_transitions)
            visualize_actor_critic(test_states, test_actions, test_q_values, actor_params, env, save_path,
                                   "Online_Interaction_Offline_Data", j)

        # TODO fix this
        alerts_plot_info = merge_dictionaries(alerts_plot_info, alert_info)

        update_pbar(pbar, stats, cfg.experiment.online_stat_keys)

    # Make Alerts Plot
    alert_trace_thresholds = alerts.get_trace_thresh()
    plot_action_value_alert(alerts_plot_info, alert_trace_thresholds, save_path)

    return online_eval
