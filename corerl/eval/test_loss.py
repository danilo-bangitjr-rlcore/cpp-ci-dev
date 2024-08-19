from pathlib import Path
import json
from corerl.eval.base_eval import BaseEval
from corerl.component.buffer.factory import init_buffer


class TestLossEval(BaseEval):
    def __init__(self, cfg, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")
        if 'eval_transitions' not in kwargs:
            raise KeyError("Missing required argument: 'eval_transitions'")

        eval_transitions = kwargs['eval_transitions']
        assert len(eval_transitions) > 0, "To use the TestLossEval, the number of passed eval transitions must be greater than 0"

        self.agent = kwargs['agent']
        self.buffer = init_buffer(cfg.buffer)
        for transition in eval_transitions:
            self.buffer.feed(transition)
        self.ensemble = cfg.ensemble
        self.test_losses = []

    def do_eval(self, **kwargs) -> None:
        batches = self.buffer.sample()
        test_loss = sum(self.agent.compute_critic_loss(batches))
        self.test_losses.append(test_loss.detach().numpy())

    def get_stats(self):
        stats = {}
        stats["test_losses"] = self.test_losses
        return stats

    def output(self, path: Path):
        stats = self.get_stats()
        return stats
