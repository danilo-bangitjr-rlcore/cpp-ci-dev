from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from corerl.configs.config import MISSING, computed, config, sanitizer
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.maybe import Maybe

if TYPE_CHECKING:
    from corerl.config import MainConfig


type list_or_single[T] = list[T] | T

@config()
class Goal:
    op: Literal['down_to', 'up_to'] = MISSING
    tag: str = MISSING
    thresh: float = MISSING


@config()
class JointGoal:
    op: Literal['and', 'or'] = MISSING
    goals: list[Goal | JointGoal] = Field(default_factory=list, min_length=1)


@config()
class Optimization:
    tags: list[str] = Field(default_factory=list, min_length=1)
    directions: list_or_single[Literal['min', 'max']] = MISSING
    weights: list[float] = MISSING

    @computed('weights')
    @classmethod
    def _weights(cls, cfg: MainConfig):
        assert cfg.pipeline.reward is not None
        opt = cfg.pipeline.reward.priorities[-1]
        assert isinstance(opt, Optimization)

        return [1.0 for _ in range(len(opt.tags))]


Priority = Goal | JointGoal | Optimization


@config()
class RewardConfig:
    priorities: list[Priority] = Field(default_factory=list, min_length=1)

    # ----------------
    # -- Validators --
    # ----------------
    @sanitizer
    def _check_optimization(self, cfg: MainConfig):
        for priority in self.priorities[:-1]:
            assert not isinstance(priority, Optimization), 'Optimization can only be applied to the last priority'

        assert isinstance(self.priorities[-1], Optimization), 'Last priority must be an optimization'

    @sanitizer
    def _check_tags_exist(self, cfg: MainConfig):
        known_tags = set(tag.name for tag in cfg.pipeline.tags)
        _assert_tags_exist(self.priorities, known_tags)

    @sanitizer
    def _check_in_range(self, cfg: MainConfig):
        for priority in self.priorities:
            _assert_tag_in_range(priority, cfg.pipeline.tags)


def _assert_tags_exist(priorities: Sequence[Priority], known_tags: set[str]):
    for priority in priorities:
        if isinstance(priority, Goal):
            assert priority.tag in known_tags

        elif isinstance(priority, Optimization):
            for tag in priority.tags:
                assert tag in known_tags

        else:
            _assert_tags_exist(priority.goals, known_tags)


def _assert_tag_in_range(priority: Priority, tag_cfgs: list[TagConfig]):
    if isinstance(priority, Goal):
        op_range = (
            Maybe.find(lambda cfg: cfg.name == priority.tag, tag_cfgs)
            .map(lambda cfg: cfg.operating_range)
        )

        lo = op_range.map(lambda rng: rng[0]).unwrap()
        hi = op_range.map(lambda rng: rng[1]).unwrap()

        assert lo is not None and priority.thresh > lo, \
            f"Goal {priority.tag} is outside of operating range [{lo}, {hi}]"
        assert hi is not None and priority.thresh < hi, \
            f"Goal {priority.tag} is outside of operating range [{lo}, {hi}]"

    elif isinstance(priority, JointGoal):
        for goal in priority.goals:
            _assert_tag_in_range(goal, tag_cfgs)
