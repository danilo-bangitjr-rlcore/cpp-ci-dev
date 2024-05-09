from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class RewardEval(BaseEval):
    def __init__(self, cfg,  **kwargs):
        self.gamma = cfg.gamma
        self.episode_steps = 0
        self.episode_return = 0
        self.return_window = cfg.return_window
        self.reward_window = cfg.reward_window
        self.returns = []
        self.rewards = []

    def do_eval(self, **kwargs) -> None:
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")
        transitions = kwargs['transitions']
        for transition in transitions:
            _, _, reward, _, done, truncate, _, _, gamma_exp = transition
            self.episode_return += reward * (self.gamma ** self.episode_steps)
            self.rewards.append(reward)
            if done or truncate:
                self.episode_steps = 0
                self.returns.append(self.episode_return)
                self.episode_return = 0
            else:
                self.episode_steps += 1

    def get_stats(self):
        stats = {
            'num_episodes': len(self.returns),
            'avg_reward': sum(self.rewards) / len(self.rewards),
        }

        if len(self.returns) > 0:
            stats['avg_return'] = sum(self.returns) / len(self.returns)
        else:
            stats['avg_return'] = 'n/a'

        if len(self.rewards) > self.reward_window:
            stats['avg_reward ({})'.format(self.reward_window)] = sum(
                self.rewards[-self.reward_window:]) / self.reward_window
        else:
            stats['avg_reward ({})'.format(self.reward_window)] = 'n/a'

        if len(self.returns) > self.return_window:
            stats['avg_return ({})'.format(self.return_window)] = sum(
                self.returns[-self.return_window:]) / self.return_window
        else:
            stats['avg_return ({})'.format(self.return_window)] = 'n/a'

        stats['returns'] = self.returns
        stats['rewards'] = self.rewards

        return stats

    def output(self, path: Path):
        stats = self.get_stats()
        return stats
