from corerl.state_constructor import sc_group
from corerl.utils.hydra import DiscriminatedUnion

def init_state_constructor(cfg: DiscriminatedUnion):
    return sc_group.dispatch(cfg)
