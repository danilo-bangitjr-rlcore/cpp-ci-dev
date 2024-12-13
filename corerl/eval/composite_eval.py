import pickle as pkl
from typing import Optional
from functools import reduce
from pathlib import Path
from corerl.eval.base_eval import BaseEval, EvalConfig
from corerl.eval.factory import init_single_evaluator


class CompositeEval(BaseEval):
    """
    This class is used when we wish to use multiple evaluators
    """

    def __init__(
        self,
        cfg: Optional[dict[str, EvalConfig]],
        eval_args: dict,
        online: bool = False,
        offline: bool = False
    ):
        self.evaluators = _instantiate_evaluators(cfg, eval_args, online=online, offline=offline)

    def do_eval(self, **kwargs) -> None:
        for evaluator in self.evaluators:
            evaluator.do_eval(**kwargs)

    def get_stats(self) -> dict:
        if len(self.evaluators) > 0:
            all_stats_list = [e.get_stats() for e in self.evaluators]
            # merge stats
            merged_stats = reduce(lambda x, y: {**x, **y}, all_stats_list)
            return merged_stats
        else:
            return {}

    def save(self, save_path: Path, prefix: str) -> None:
        stats = self.get_stats()

        stats_path = save_path / "{}_eval.pkl".format(prefix)
        with open(stats_path, "wb") as f:
            pkl.dump(stats, f)


def _instantiate_evaluators(
    eval_cfg: Optional[dict[str, EvalConfig]],
    eval_args,
    online=False,
    offline=False
) -> list[BaseEval]:
    assert online != offline, "Set either online or offline arg to True"
    evaluators = []
    if eval_cfg is not None:
        for eval_type in eval_cfg.keys():
            eval_type_cfg = eval_cfg[eval_type]
            # check if the mode of running for the evaluator matches the offline/online flag
            if (online and eval_cfg[eval_type].online_eval) or (offline and eval_cfg[eval_type].offline_eval):
                evaluators.append(init_single_evaluator(eval_type_cfg, eval_args))

    return evaluators
