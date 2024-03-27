from nltk.draw import cfg

import critic_networks




def init_critic_network(name, device, input_dim, hidden_units, output_dim, activation, layer_init, layer_norm,
                        ensemble):
    hidden_units = [i for i in hidden_units if i > 0]
    if name == "FC":
        return EnsembleCritic(device, input_dim, hidden_units, output_dim, ensemble=ensemble, activation=activation, \
                              head_activation="None", init=layer_init, layer_norm=layer_norm)
    else:
        raise NotImplementedError


def init_custom_network(name, device, input_dim, hidden_units, output_dim, activation, head_activation, layer_init,
                        layer_norm):
    hidden_units = [i for i in hidden_units if i > 0]
    if name == "FC":
        return FC(device, input_dim, hidden_units, output_dim,
                  activation=activation, head_activation=head_activation, init=layer_init, layer_norm=layer_norm)
    elif name == "Softmax":
        return Softmax(device, input_dim, hidden_units, output_dim, activation=activation, init=layer_init,
                       layer_norm=layer_norm)
    elif name == "RndLinearUncertainty":
        return RndLinearUncertainty(device, input_dim, hidden_units, output_dim, activation=activation, init=layer_init,
                                    layer_norm=layer_norm)
    else:
        raise NotImplementedError
