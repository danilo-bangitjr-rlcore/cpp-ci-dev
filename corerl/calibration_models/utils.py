import numpy as np
from corerl.component.network.utils import tensor


class Trajectory:
    def __init__(self, start_sc, is_test=False):
        self.start_sc = start_sc
        self.transitions = []
        self.endo_vars = []
        self.exo_vars = []
        self.is_test = is_test

    def add_transition(self, state_transition: tuple) -> None:
        self.transitions.append(state_transition)

    def add_endo_var(self, endo_var: np.ndarray) -> None:
        self.endo_vars.append(endo_var)

    def add_exo_var(self, exo_var: np.ndarray) -> None:
        self.exo_vars.append(exo_var)

    def num_transitions(self):
        return len(self.transitions)


def prepare_obs_transition(transition):
    return tensor(transition[0]), tensor(transition[1]), transition[3]
