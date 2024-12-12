from corerl.data import register, tc_group
from corerl.state_constructor.base import BaseStateConstructor
from corerl.utils.hydra import DiscriminatedUnion


def init_transition_creator(cfg: DiscriminatedUnion, state_constuctor: BaseStateConstructor):
    register()
    return tc_group.dispatch(cfg, state_constuctor)
