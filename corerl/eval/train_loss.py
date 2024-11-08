from corerl.eval.base_eval import BaseEval, EvalConfig
from corerl.utils.hydra import config, interpolate


@config('train_loss', group='eval')
class TrainLossConfig(EvalConfig):
    name: str = 'train_loss'

    ensemble: bool = interpolate('${agent.critic.critic_network.ensemble}')
    offline_eval: bool = True
    online_eval: bool = False


class TrainLossEval(BaseEval):
    def __init__(self, cfg: TrainLossConfig, **kwargs):
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
