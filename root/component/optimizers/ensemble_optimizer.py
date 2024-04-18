class EnsembleOptimizer:
    def __init__(self, individual_optim, param, lr, kwargs):
        self.optim = [
            individual_optim(list(p), lr, **kwargs) for p in param
        ]

    def zero_grad(self):
        for opt in self.optim:
            opt.zero_grad()
        return

    def step(self):
        for opt in self.optim:
            opt.step()
        return

    def state_dict(self):
        return [opt.state_dict() for opt in self.optim]

    def load_state_dict(self, state_dict_lst):
        for opt, sd in zip(self.optim, state_dict_lst):
            opt.load_state_dict(sd)
        return

    @property
    def param_groups(self):
        pg = []
        for opt in self.optim:
            # assert type(opt.param_groups) == list
            pg += opt.param_groups
            # pg.append(opt.param_groups)
        return pg
