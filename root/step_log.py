import pickle as pkl

LOG = None

class StepLog(dict):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def __setitem__(self, key, value):
        key = key.upper()
        super().__setitem__(key, value)

    def save(self):
        with open(self.save_path / 'step-{}'.format(self.step), 'wb') as f:
            pkl.dump(self, f)

    def increment(self):
        self.step += 1


def init_step_log(save_path):
    global LOG
    LOG = StepLog(save_path)
