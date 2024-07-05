
from corerl.component.buffer.factory import init_buffer


class KNNCalibrationModel:

    def __init__(self, cfg, train_info: dict):
        self.test_trajectories = train_info['test_trajectories_cm']
        train_transitions = train_info['train_transitions_cm']
        self.reward_func = train_info['reward_func']
        self.normalizer = train_info['normalizer']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)
        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds



