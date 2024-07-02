from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class ActionEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        if 'env' not in kwargs:
            raise KeyError("Missing required argument: 'env'")
        if 'transition_normalizer' not in kwargs:
            raise KeyError("Missing required argument: 'transition_normalizer'")

        self.env = kwargs['env']
        self.transition_normalizer = kwargs['transition_normalizer']
        self.action_names = self.env.action_names
        self.actions = {}
        for action_name in self.action_names:
            self.actions[action_name] = []

    def do_eval(self, **kwargs) -> None:
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")

        transitions = kwargs['transitions']

        for transition in transitions:
            transition_copy = self.transition_normalizer.denormalize(transition)
            action = transition_copy.action
            for i in range(len(action)):
                self.actions[self.action_names[i]].append(action[i].item())

    def get_stats(self):
        stats = {}
        stats["raw_actions"] = self.actions
        return stats

    def output(self, path: Path):
        stats = self.get_stats()
        return stats
