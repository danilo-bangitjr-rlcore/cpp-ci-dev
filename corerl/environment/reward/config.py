from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Callable, Literal

from pydantic import Field

from corerl.configs.config import MISSING, computed, config, post_processor
from corerl.data_pipeline.tag_config import TagConfig
from corerl.utils.maybe import Maybe
from corerl.utils.sympy import is_affine, is_expression, to_sympy

if TYPE_CHECKING:
    from corerl.config import MainConfig


type list_or_single[T] = list[T] | T

@config()
class Goal:
    op: Literal['down_to', 'up_to'] = MISSING
    tag: str = MISSING
    thresh: float | str = MISSING

    # Exclude these fields from openapi.json because Callable causes problems in CoreIO
    # These attributes are set in the post processor
    thresh_func: Annotated[Callable[..., float] | None, Field(exclude=True)] = None
    thresh_tags: Annotated[list[str] | None, Field(exclude=True)] = None

    @post_processor
    def _parse_sympy(self, cfg: MainConfig):
        if isinstance(self.thresh, str) and is_expression(self.thresh):
            expression, self.thresh_func, self.thresh_tags = to_sympy(self.thresh)
            assert is_affine(expression)


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
    @post_processor
    def _check_optimization(self, cfg: MainConfig):
        for priority in self.priorities[:-1]:
            assert not isinstance(priority, Optimization), 'Optimization can only be applied to the last priority'

        assert isinstance(self.priorities[-1], Optimization), 'Last priority must be an optimization'

    @post_processor
    def _check_tags_exist(self, cfg: MainConfig):
        known_tags = set(tag.name for tag in cfg.pipeline.tags)
        _assert_tags_exist(self.priorities, known_tags)

    @post_processor
    def _check_in_range(self, cfg: MainConfig):
        for priority in self.priorities:
            _assert_tag_in_range(priority, cfg.pipeline.tags)


def _assert_tags_exist(priorities: Sequence[Priority], known_tags: set[str]):
    for priority in priorities:
        if isinstance(priority, Goal):
            assert priority.tag in known_tags

            if isinstance(priority.thresh, str):
                # If thresh is an expression
                if priority.thresh_func is not None:
                    assert priority.thresh_tags is not None
                    for thresh_tag in priority.thresh_tags:
                        assert thresh_tag in known_tags
                        assert thresh_tag != priority.tag

                # If thresh is a tag
                else:
                    assert priority.thresh in known_tags
                    # A tag can't be its own threshold
                    assert priority.tag != priority.thresh

        elif isinstance(priority, Optimization):
            for tag in priority.tags:
                assert tag in known_tags

        else:
            _assert_tags_exist(priority.goals, known_tags)


def _assert_tag_in_range(priority: Priority, tag_cfgs: list[TagConfig]):
    if isinstance(priority, Goal) and isinstance(priority.thresh, float):
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
