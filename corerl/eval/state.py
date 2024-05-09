from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class StateEval(BaseEval):
    def __init__(self, cfg,  **kwargs):
        self.start = True
        self.states = []

    def do_eval(self, **kwargs) -> None:
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")
        transitions = kwargs['transitions']
        for transition in transitions:
            state, _, _, next_state, _, _, _, _ = transition
            if self.start:
                self.start = False
                for state_elem in state:
                    self.states.append([state_elem])

            for i, state_elem in enumerate(next_state):
                self.states[i].append(state_elem)

    def get_stats(self):
        return {'states': self.states}

    def output(self, path: Path):
        stats = self.get_stats()

        with open(path, 'w') as f:
            json.dump(stats, f)
