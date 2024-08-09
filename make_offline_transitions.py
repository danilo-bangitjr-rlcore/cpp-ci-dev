import hydra
import numpy as np
import torch
import random
import pandas as pd

from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path

from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.interaction.factory import init_interaction
from corerl.utils.device import device
from corerl.data_loaders.factory import init_data_loader
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.data_loaders.utils import train_test_split
from corerl.eval.composite_eval import CompositeEval
from corerl.utils.plotting import make_plots

import corerl.utils.freezer as fr
import main_utils as utils


def add_time_column(df):
    base = pd.Timestamp("2000-01-01")  # You can choose any base date here
    df['time'] = base + pd.to_timedelta(range(len(df)), unit='s')
    return df


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

    normalizer = ObsTransitionNormalizer(cfg.normalizer, env)
    composite_alert = CompositeAlert(cfg.alerts, alert_args)
    transition_creator = AnytimeTransitionCreator(cfg.transition_creator, composite_alert)
    interaction = init_interaction(cfg.interaction, env, sc, composite_alert,
                                   transition_creator, normalizer)

    # Instantiate online evaluators
    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    action = agent.get_action(state)  # initial action
    all_transitions = []

    observations = []
    actions = []

    print('Starting online training...')
    for _ in pbar:
        transitions, train_transitions, alert_info_list, env_info_list = interaction.step(action)

        for transition in transitions:  # agent_train_transitions:
            agent.update_buffer(transition)
            observations.append(transition.obs)
            actions.append(transition.action)

        if agent.buffer.size > 0:
            agent.update()

        if len(transitions) > 0:
            online_eval_args = {
                'agent': agent,
                'env': env,
                'transitions': transitions}

            online_eval.do_eval(**online_eval_args)
            stats = online_eval.get_stats()
            utils.update_pbar(pbar, stats, cfg.experiment.online_stat_keys)

            terminated = transitions[-1].terminated
            truncated = transitions[-1].truncate
            if terminated or truncated:
                state, _ = interaction.reset()
            else:
                state = transitions[-1].next_state
            action = agent.get_action(state)

    observations = np.array(observations)
    actions = [np.zeros_like(action)] + actions[:-1]
    actions = np.array(actions)

    dataset = np.concatenate((observations, actions), axis=1)
    colnames = cfg.env.obs_names + cfg.env.action_names

    df = pd.DataFrame(dataset, columns=colnames)
    df = add_time_column(df)

    print("Generated df:")
    print(df.head())

    output_path = Path(cfg.offline_data.output_path) / 'csv'
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'data.csv')


if __name__ == "__main__":
    main()
