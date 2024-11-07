"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import numpy as np
import torch

from corerl.eval.base_eval import BaseEval, EvalConfig
from corerl.data.data import TransitionBatch
from corerl.component.network.utils import to_np
from corerl.utils.hydra import config


@config('q_estimation', group='eval')
class QEstimationConfig(EvalConfig):
    name: str = 'q_estimation'
    offline_eval: bool = True
    online_eval: bool = True


class QEstimation(BaseEval):
    def __init__(self, cfg: QEstimationConfig, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")

        self.agent = kwargs['agent']
        assert hasattr(self.agent, 'q_critic'), "Agent must have a q_critic"
        self.qs_max: list[list[float]] = []
        self.qs_min: list[list[float]] = []
        self.qs_median: list[list[float]] = []
        self.qs_avg: list[list[float]] = []
        self.ens_stes: list[list[float]] = []

    def do_eval(self, **kwargs) -> None:
        batches = self.agent.critic_buffer.sample()
        batch = batches[0]
        q_max, q_min, q_med, q_avg, ens_ste = self._estimate_q(batch)
        self.qs_max.append(q_max)
        self.qs_min.append(q_min)
        self.qs_median.append(q_med)
        self.qs_avg.append(q_avg)
        self.ens_stes.append(ens_ste)

    def _estimate_q(
        self,
        batch: TransitionBatch,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        state_batch = batch.state
        action_batch = batch.action

        q, qs = self.agent.q_critic.get_qs([state_batch], [action_batch], with_grad=False)
        q_max = torch.max(q).view(-1)
        q_min = torch.min(q).view(-1)
        q_median = torch.median(q).view(-1)
        q_avg = torch.mean(q).view(-1)
        ens_ste = torch.std(qs, dim=1).mean() / np.sqrt(qs.size()[1])
        return (to_np(q_max).tolist(), to_np(q_min).tolist(), to_np(q_median).tolist(),
                to_np(q_avg).tolist(), to_np(ens_ste).tolist())

    def get_stats(self) -> dict:
        stats = {
            'q_estimation_max': self.qs_max,
            'q_estimation_min': self.qs_min,
            'q_estimation_median': self.qs_median,
            'q_estimation_avg': self.qs_avg,
            'ensemble_ste': self.ens_stes,
        }
        return stats
