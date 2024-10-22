from corerl.eval.base_eval import BaseEval


class EndoObsEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        if 'env' not in kwargs:
            raise KeyError("Missing required argument: 'env'")
        if 'transition_normalizer' not in kwargs:
            raise KeyError("Missing required argument: 'transition_normalizer'")

        self.env = kwargs['env']
        self.transition_normalizer = kwargs['transition_normalizer']
        self.endo_obs_names = self.env.endo_obs_names
        self.endo_inds = self.env.endo_inds
        self.raw_endo_obs = {}
        for col_name in self.endo_obs_names:
            self.raw_endo_obs[col_name] = []

    def do_eval(self, **kwargs) -> None:
        if 'transitions' not in kwargs:
            raise KeyError("Missing required argument: 'transitions'")

        transitions = kwargs['transitions']

        for transition in transitions:
            transition_copy = self.transition_normalizer.denormalize(transition)
            curr_obs = transition_copy.obs

            for i in range(len(self.endo_obs_names)):
                col_name = self.endo_obs_names[i]
                ind = self.endo_inds[i]
                self.raw_endo_obs[col_name].append(curr_obs[ind].item())

    def get_stats(self):
        stats = {}
        stats["raw_endo_obs"] = self.raw_endo_obs
        return stats
