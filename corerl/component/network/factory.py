from corerl.component.network.networks import EnsembleNetwork


def init_target_network(target: EnsembleNetwork, original: EnsembleNetwork):
    target.load_state_dict(original.state_dict())
    return target
