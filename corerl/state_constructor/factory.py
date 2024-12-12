from corerl.state_constructor import sc_group, register
from corerl.utils.hydra import DiscriminatedUnion


def init_state_constructor(cfg: DiscriminatedUnion):
    register()
    return sc_group.dispatch(cfg)
