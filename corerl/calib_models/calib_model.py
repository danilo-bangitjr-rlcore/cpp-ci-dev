from corerl.component.buffer.factory import init_buffer

class SimpleCalibrationModel:
    def __init__(self, cfg, train_transitions, test_transitions):
        self.train_buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_buffer.load(train_transitions)
        self.test_buffer.load(test_transitions)

        self.train_itr = cfg.train_tir

    def train(self):
        pass











