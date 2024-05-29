import numpy as np
from corerl.component.network.utils import tensor
from corerl.data import Transition

class Trajectory:
    def __init__(self, start_sc, is_test=False):
        self.start_sc = start_sc
        self.transitions = []  # usual (S, A, R, S' ... )
        self.endo_vars = []  # sequence of endo variables, one for each transition
        self.exo_vars = []  # sequence of exogenous variables, one for each transition
        self.is_test = is_test  # whether or not this is a test trajectory

    def add_transition(self, state_transition: Transition) -> None:
        self.transitions.append(state_transition)

    def add_endo_var(self, endo_var: np.ndarray) -> None:
        self.endo_vars.append(endo_var)

    def add_exo_var(self, exo_var: np.ndarray) -> None:
        self.exo_vars.append(exo_var)

    def add_exo_var(self, exo_var: np.ndarray) -> None:
        self.exo_vars.append(exo_var)

    @property
    def num_transitions(self):
        return len(self.transitions)

    @property
    def endo_dim(self):
        return len(self.endo_vars[0])

    @property
    def exo_dim(self):
        return len(self.exo_vars[0])

    @property
    def action_dim(self):
        return len(self.transitions[0][1])


def prepare_obs_transition(transition):
    return tensor(transition[0]), tensor(transition[1]), transition[3]
