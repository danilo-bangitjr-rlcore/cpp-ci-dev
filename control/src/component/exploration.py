import torch
import torch.nn as nn

from src.network.torch_utils import clone_model_0to1, clone_gradient, move_gradient_to_network
from src.network.factory import init_policy_network, init_critic_network, init_optimizer, init_custom_network


class RndNetworkExplore:
    def __init__(self, device, state_dim, action_dim, hidden_size, activation):

        # Networks for uncertainty measure
        self.ftrue0 = init_custom_network("RndLinearUncertainty", device, state_dim+action_dim,
                                          hidden_size, state_dim+action_dim, activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.ftrue1 = init_custom_network("RndLinearUncertainty", device, state_dim+action_dim,
                                          hidden_size, state_dim+action_dim, activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus0 = init_custom_network("RndLinearUncertainty", device, state_dim+action_dim,
                                          hidden_size, state_dim+action_dim, activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus1 = init_custom_network("RndLinearUncertainty", device, state_dim+action_dim,
                                          hidden_size, state_dim+action_dim, activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus0_copy = init_custom_network("RndLinearUncertainty", device, state_dim+action_dim,
                                          hidden_size, state_dim+action_dim, activation, "None",
                                          "Xavier/1", layer_norm=False)
        self.fbonus1_copy = init_custom_network("RndLinearUncertainty", device, state_dim+action_dim,
                                          hidden_size, state_dim+action_dim, activation, "None",
                                          "Xavier/1", layer_norm=False)

        # Ensure the nonlinear net between learning net and target net are the same
        clone_model_0to1(self.ftrue0.random_network, self.fbonus0.random_network)
        clone_model_0to1(self.ftrue1.random_network, self.fbonus1.random_network)
        clone_model_0to1(self.fbonus0, self.fbonus0_copy)
        clone_model_0to1(self.fbonus1, self.fbonus1_copy)

    def get_networks(self):
        return self.fbonus0, self.fbonus1, self.fbonus0_copy, self.fbonus1_copy

    def explore_bonus_loss(self, state, action, reward, next_state, next_action, mask, gamma):
        in_ = torch.concat((state, action), dim=1)
        in_p1 = torch.concat((next_state, action), dim=1)
        with torch.no_grad():
            true0_t, _ = self.ftrue0(in_)
            true1_t, _ = self.ftrue1(in_)
            true0_tp1, _ = self.ftrue0(in_p1)
            true1_tp1, _ = self.ftrue1(in_p1)
        reward0 = true0_t - mask * gamma * true0_tp1
        reward1 = true1_t - mask * gamma * true1_tp1

        pred0, _ = self.fbonus0(in_)
        pred1, _ = self.fbonus1(in_)
        with torch.no_grad():
            target0 = reward0.detach() + mask * gamma * self.fbonus0(in_p1)[0] #true0_t #
            target1 = reward1.detach() + mask * gamma * self.fbonus1(in_p1)[0] #true1_t #

        # TODO: Include the reward and next state in training, for the stochasity (how?)
        loss0 = nn.functional.mse_loss(pred0, target0)
        loss1 = nn.functional.mse_loss(pred1, target1)
        return [loss0, loss1]

    def explore_bonus_eval(self, state, action):
        in_ = torch.concat((state, action), dim=1)
        with torch.no_grad():
            pred0, _ = self.fbonus0(in_)
            pred1, _ = self.fbonus1(in_)
            true0, _ = self.ftrue0(in_)
            true1, _ = self.ftrue1(in_)
        b, _ = torch.max(torch.concat([torch.abs(pred0 - true0), torch.abs(pred1 - true1)], dim=1),
                         dim=1, keepdim=True)
        return b.detach()

