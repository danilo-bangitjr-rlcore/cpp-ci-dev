from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class StateEval(BaseEval):
    def __init__(self, cfg,  **kwargs):
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")
        self.start = True
        self.states: list[list[float]] = []

    def do_eval(self, **kwargs) -> None:
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")
        transitions = kwargs['transitions']
        for transition in transitions:
            state, next_state = transition[0], transition[3]
            if self.start:
                self.start = False
                for state_elem in state:
                    self.states.append([state_elem])

            for i, state_elem in enumerate(next_state):
                self.states[i].append(state_elem)

    def get_stats(self):
        return {'states': self.states}

