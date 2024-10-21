from corerl.eval.base_eval import BaseEval


class RewardEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        self.gamma = cfg.gamma
        self.episode_steps = 0
        self.episode_return = 0
        self.return_window = cfg.return_window
        self.reward_window = cfg.reward_window
        self.returns = []
        self.rewards = []
        self.reward_sum = 0
        self.reward_sums = []

    def do_eval(self, **kwargs) -> None:
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")

        transitions = kwargs['transitions']
        for transition in transitions:
            reward = transition.reward
            terminated = transition.terminated
            self.reward_sum += reward
            self.reward_sums.append(self.reward_sum)
            self.episode_return += reward * (self.gamma ** self.episode_steps)
            self.rewards.append(reward)
            if terminated:
                self.episode_steps = 0
                self.returns.append(self.episode_return)
                self.episode_return = 0
            else:
                self.episode_steps += 1

    def get_stats(self):
        stats = {'num_episodes': len(self.returns)}
        if len(self.rewards) > 0:
            stats['avg_reward'] = sum(self.rewards) / len(self.rewards)
        else:
            stats['avg_reward'] = 'n/a'

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

        stats['rewards'] = self.rewards
        stats['returns'] = self.returns
        stats['reward_sums'] = self.reward_sums

        return stats
