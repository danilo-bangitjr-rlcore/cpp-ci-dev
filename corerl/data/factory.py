from corerl.data.transition_creator import tc_group
from corerl.state_constructor.base import BaseStateConstructor
from corerl.utils.hydra import DiscriminatedUnion

def init_transition_creator(cfg: DiscriminatedUnion, state_constuctor: BaseStateConstructor):
    return tc_group.dispatch(cfg, state_constuctor)
