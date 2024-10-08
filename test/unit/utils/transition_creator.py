import pytest
from omegaconf import DictConfig

from corerl.data.transition_creator import AnytimeTransitionCreator, RegularRLTransitionCreator



@pytest.fixture
def anytime_transition_creator():
    pass

@pytest.fixture
def anytime_transition_creator():
    pass
    # cfg_d = {
    #     'steps_per_decision': 3,
    #     'n_step': 0,
    #     'gamma': 0.9,
    #     'transition_kind': 'anytime'
    # }
    # cfg = DictConfig(cfg_d)

