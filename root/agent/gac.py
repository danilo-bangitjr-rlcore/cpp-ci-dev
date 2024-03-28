import torch
from base import BaseAgent

class GAC(BaseAgent):
    def __init__(self, cfg):
        self.tau = cfg.tau
        self.rho = cfg.rho
        self.rho_proposal = self.rho * cfg.prop_rho_mult
        self.num_samples = cfg.n
        self.average_entropy = cfg.average_entropy


        # TODO implement these
        self.actor = None
        self.critic = None
        self.proposal = None
