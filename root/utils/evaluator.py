from pathlib import Path
import json


class Evaluator:
    def __init__(self, cfg):
        self.gamma = cfg.gamma
        self.episode_steps = 0
        self.episode_return = 0
        self.return_window = cfg.return_window
        self.reward_window = cfg.reward_window
        self.returns = []
        self.rewards = []

    def update(self, transition: tuple) -> None:
        state, action, reward, next_state, done, truncate = transition
        self.episode_return += reward * (self.gamma ** self.episode_steps)
        self.rewards.append(reward)
        if done:
            self.episode_steps = 0
            self.returns.append(self.episode_return)
            self.episode_return = 0
        else:
            self.episode_steps += 1

    def get_stats(self):
        stats = {'num_episodes': len(self.returns),
                 'avg_reward': sum(self.rewards) / len(self.rewards),
                 'avg_return': sum(self.returns) / len(self.returns),
                 }

        if len(self.rewards) > self.reward_window:
            stats['avg_reward ({})'.format(self.reward_window)] = sum(self.rewards[-self.reward_window:]) / self.reward_window
        else:
            stats['avg_reward ({})'.format(self.reward_window)] = 'n/a'

        if len(self.returns) > self.return_window:
            stats['avg_return ({})'.format(self.return_window)] = sum(self.returns[-self.return_window:]) / self.return_window
        else:
            stats['avg_return ({})'.format(self.return_window)] = 'n/a'

        return stats

    def output(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.get_stats(), f)
