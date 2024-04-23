import pickle as pkl

freezer = None


class Freezer:
    def __init__(self, save_path):
        super().__init__()
        self.dict = dict()
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def __setitem__(self, key, value):
        self.dict[key] = value

    def save(self):
        with open(self.save_path / 'step-{}.pkl'.format(self.step), 'wb') as f:
            pkl.dump(self.dict, f)

    def increment(self):
        self.step += 1

    def clear(self):
        self.dict = dict()


def init_freezer(save_path):
    global freezer
    freezer = Freezer(save_path)
