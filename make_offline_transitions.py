import hydra
import numpy as np
import torch
import random
import logging
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from corerl.config import MainConfig
from corerl.agent.factory import init_agent
from corerl.environment.factory import init_environment
from corerl.state_constructor.factory import init_state_constructor
from corerl.interaction.factory import init_interaction
from corerl.utils.device import device
from corerl.data.obs_normalizer import ObsTransitionNormalizer
from corerl.data.factory import init_transition_creator
from corerl.eval.composite_eval import CompositeEval

import corerl.utils.freezer as fr
import main_utils as utils

log = logging.getLogger(__name__)


def add_time_column(df):
    base = pd.Timestamp("2000-01-01")  # You can choose any base date here
    df['time'] = base + pd.to_timedelta(range(len(df)), unit='s')
    return df

def output_to_df(cfg, observations, actions):
    initial_action = actions[0]
    observations = np.array(observations)
    actions = [np.zeros_like(initial_action)] + actions[:-1]
    actions = np.array(actions)

    dataset = np.concatenate((observations, actions), axis=1)
    colnames = cfg.env.obs_names + cfg.env.action_names

    df = pd.DataFrame(dataset, columns=colnames)
    df = add_time_column(df)

    log.info(f'Generated df: \n{df.head()}')

    output_path = Path(cfg.offline_data.output_path) / 'csv'
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / 'data.csv')


@hydra.main(version_base=None, config_name='config', config_path="config/")
def main(cfg: MainConfig) -> dict:
    save_path = utils.prepare_save_dir(cfg)
    fr.init_freezer(save_path / 'logs')
    device.update_device(cfg.experiment.device)

    # set the random seeds
    seed = cfg.experiment.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = init_environment(cfg.env)
    sc = init_state_constructor(cfg.state_constructor)
    state_dim, action_dim = utils.get_state_action_dim(env, sc)
    log.debug(f"State Dim: {state_dim}, action dim: {action_dim}")
    agent = init_agent(cfg.agent, state_dim, action_dim)

    normalizer = ObsTransitionNormalizer(cfg.normalizer, env)
    transition_creator = init_transition_creator(cfg.agent_transition_creator, sc)
    interaction = init_interaction(cfg.interaction, env, sc, transition_creator, normalizer)

    # Instantiate online evaluators
    online_eval_args = {
        'agent': agent
    }
    online_eval = CompositeEval(cfg.eval, online_eval_args, online=True)

    max_steps = cfg.experiment.max_steps
    pbar = tqdm(range(max_steps))
    state, info = interaction.reset()
    action = agent.get_action(state)

    observations = []
    actions = []

    log.info('Starting online training...')
    alert_info_list = []
    for _ in pbar:
        transitions, agent_train_transitions, _, _, _, env_info = interaction.step(action)

        for transition in transitions:
            observations.append(transition.obs)
            actions.append(transition.action)

        for transition in agent_train_transitions:
            agent.update_buffer(transition)

        agent.update()

        if len(transitions) > 0:
            # logging + evaluation
            # union of the information needed by all evaluators
            online_eval_args = {
                'agent': agent,
                'env': env,
                'transitions': transitions,
                'alert_info_list': alert_info_list
            }

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

    output_to_df(cfg, observations, actions)


if __name__ == "__main__":
    main()
