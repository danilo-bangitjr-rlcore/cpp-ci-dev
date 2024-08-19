from corerl.eval.base_eval import BaseEval
from pathlib import Path
import json


class TrainLossEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        self.ensemble = cfg.ensemble
        self.train_losses = []

    def do_eval(self, **kwargs) -> None:
        if 'train_loss' not in kwargs:
            raise KeyError("Missing required argument: 'train_loss'")

        train_loss = kwargs['train_loss']
        self.train_losses.append(train_loss.cpu().detach().numpy())

    def get_stats(self):
        stats = {}
        stats["train_losses"] = self.train_losses
        return stats

    def output(self, path: Path):
        stats = self.get_stats()
        return stats
