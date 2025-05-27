import logging
from functools import partial

import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig, get_tag_bounds
from corerl.environment.reward.config import Goal, JointGoal, Optimization, RewardConfig
from corerl.utils.math import put_in_range
from corerl.utils.maybe import Maybe

logger = logging.getLogger(__name__)

class GoalConstructor:
    def __init__(self, reward_cfg: RewardConfig, tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
        self._cfg = reward_cfg
        self._tag_cfgs = tag_cfgs
        self._prep_stage = prep_stage
        self.ignore_oob_tags_in_compound_goals = reward_cfg.ignore_oob_tags_in_compound_goals

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        # denormalize all tags before checking constraint violations
        pf.rewards = self.denormalize_tags(pf.data)

        rewards = []
        for (_, row_series) in pf.rewards.iterrows():
            row = row_series.to_frame().transpose()
            active_idx = self._find_active_priority_idx(row)
            priority = self._cfg.priorities[active_idx]
            if not isinstance(priority, Optimization):
                violation_percent = self._priority_violation_percent(priority, row)

                # break [-1, -0.5] into num_priorities-1 buckets
                num_buckets = len(self._cfg.priorities) - 1
                buckets = np.linspace(-1, -0.5, num_buckets + 1)
                r = put_in_range(-violation_percent, old_range=(-1, 0), new_range=buckets[active_idx:active_idx+1])

            else:
                opt = self._avg_optimization_violation(priority, row)
                r = put_in_range(opt, old_range=(-1, 0), new_range=(-0.5, 0))

            rewards.append(r)

        pf.rewards = pd.DataFrame({
            'reward': rewards
        }, index=pf.data.index)

        return pf


    def _avg_optimization_violation(self, optimization: Optimization, row: pd.DataFrame):
        """
        When in the final optimization priority, the reward is defined as the (weighted)
        average of violation percentages for each tag being optimized. The percentage is
        defined as the "percent of the way from the lower bound to the upper bound" where
        minimizing is distance from the lower bound and maximizing is distance from the upper bound.
        """
        directions = optimization.directions
        if isinstance(optimization.directions, str):
            directions = [optimization.directions] * len(optimization.tags)

        vals = np.array([
            self._tag_optimization_violation_delta(tag, dir, row)
            for tag, dir in zip(optimization.tags, directions, strict=True)
        ])

        # produce a weighted average over the optimization deltas
        weights = np.array(optimization.weights)
        weights /= weights.sum()
        return -weights.dot(vals)

    def _tag_optimization_violation_delta(self, tag: str, dir: str, row: pd.DataFrame):
        """
        Get back the degree of violation from the target bound for each tag normalized by the tag's total range.
        If minimizing, this is distance from the lower bound.
        If maximizing, this is distance from the upper bound.
        """
        bounds = (
            Maybe.find(lambda cfg: cfg.name == tag, self._tag_cfgs)
            .map(partial(get_tag_bounds, row=row))
            .expect(f'Was unable to find tag config for tag: {tag}')
        )
        lo = bounds[0].expect(f'Was unable to find a lower bound for tag: {tag}')
        hi = bounds[1].expect(f'Was unable to find an upper bound for tag: {tag}')

        x: float = row[tag].to_numpy()[0]
        if dir == 'max':
            return (hi - x) / (hi - lo)

        return (x - lo) / (hi - lo)


    def _row_is_out_of_operating_range(self, goal: Goal, row: pd.DataFrame) -> bool:
        """
        Check if the row goal is out of bounds with respect to the tag configuration operating range.
        """
        tag_config = (
            Maybe.find(lambda cfg: cfg.name == goal.tag, self._tag_cfgs)
            .expect(f'Was unable to find tag config for tag: {goal.tag}')
        )
        if tag_config.operating_range is None:
            # no operating range, so no out of bounds
            return False

        x = row[goal.tag].to_numpy()[0]
        if np.isnan(x):
            return True

        [op_lo, op_high] = tag_config.operating_range
        if op_lo is not None and x < op_lo:
            return True

        if op_high is not None and x > op_high:
            return True

        return False


    def _priority_violation_percent(self, priority: Goal | JointGoal, row: pd.DataFrame) -> float:
        """
        Because a priority can be composed of an arbitrary tree of Goals,
        we have to recursively loop over the priority tree to calculate the violation percent.

        If taking the AND between goals:
          - E.g. min tag1 -> 0.5 AND max tag2 -> 0.9
        then the violation percent is the max violation between the two goals, indicating a
        pressure to fix the worst violation.

        If taking the OR between goals:
          - E.g. min tag1 -> 0.5 OR max tag2 -> 0.9
        then the violation percent is the min violation between the two goals, indicating a
        pressure to focus on whichever goal is closest to being achieved.

        If oob_tags_in_compound_goals is enabled, we drop tags that are out of operating range bounds.
        The nuance here is that we treat or/and differently.
        Dropping a tag from an AND is equivalent to satisfied subgoal (AND-ing with True identity).
        If the tag is part of an OR compound goal, it should be failing (OR-ing with Fail identity).
        """
        if isinstance(priority, JointGoal):
            violation_percents = [
                (goal, self._priority_violation_percent(goal, row))
                for goal in priority.goals
            ]

            for idx, (goal, _) in enumerate(violation_percents):
                if (
                    self.ignore_oob_tags_in_compound_goals
                    and isinstance(goal, Goal)
                    and self._row_is_out_of_operating_range(goal, row)
                ):
                    if priority.op == 'and':
                        # drop the tag from the AND
                        violation_percents[idx] = (goal, 0)
                        logger.warning(
                            f"Goal {goal.tag} is out of operating range. Setting violation percent to 0 for AND goal."
                        )
                    else:
                        # drop the tag from the OR
                        violation_percents[idx] = (goal, 1)
                        logger.warning(
                            f"Goal {goal.tag} is out of operating range. Setting violation percent to 1 for OR goal."
                        )

            if priority.op == 'and':
                return np.max([pct[1] for pct in violation_percents])
            return np.min([pct[1] for pct in violation_percents])

        return self._goal_violation_percent(priority, row)


    def _goal_violation_percent(self, goal: Goal, row: pd.DataFrame):
        """
        To produce a sloped reward for violating a goal, we need the degree of violation.
        This degree of violation is expressed as a percent of the range of the tag.

        If we are trying to decrease a tag, then the percent of violation is from
        the tag's upper bound to the target threshold. If we are increasing a tag,
        then the percent violoation from tag's lower bound to the target threshold.
        """

        # Threshold is float
        if isinstance(goal.thresh, float):
            thresh = goal.thresh

        # Threshold is tag
        elif goal.thresh_func is None:
            thresh = row[goal.thresh].item()

        # Threshold is expression
        else:
            assert goal.thresh_tags is not None # Assertion for pyright
            values = [row[thresh_tag].item() for thresh_tag in goal.thresh_tags]
            thresh = goal.thresh_func(*values)


        bounds = (
            Maybe.find(lambda cfg: cfg.name == goal.tag, self._tag_cfgs)
            .map(partial(get_tag_bounds, row=row))
            .expect(f'Was unable to find tag config for tag: {goal.tag}')
        )

        x: float = row[goal.tag].to_numpy()[0]
        if goal.op == 'down_to':
            hi = bounds[1].expect(f'Was unable to find an upper bound for tag: {goal.tag}')
            delta = x - thresh
            return delta / (hi - thresh)

        lo = bounds[0].expect(f'Was unable to find a lower bound for tag: {goal.tag}')
        delta = thresh - x
        return delta / (thresh - lo)


    def _find_active_priority_idx(self, row: pd.DataFrame):
        """
        Active priority is the first priority that is not satisfied.
        """
        for i, priority in enumerate(self._cfg.priorities):
            if isinstance(priority, Optimization):
                return i

            if not self._priority_is_satisfied(priority, row):
                return i

        raise ValueError('No active goal found')


    def _priority_is_satisfied(self, priority: Goal | JointGoal, row: pd.DataFrame) -> bool:
        """
        A priority is either a simple Goal or a nested JointGoal. This function scans over
        all arbitrarily nested goals and combines joint goals using the given op.
        """
        if isinstance(priority, Goal):
            return self._goal_is_satisfied(priority, row)

        if priority.op == 'and':
            return all([self._priority_is_satisfied(goal, row) for goal in priority.goals])

        return any([self._priority_is_satisfied(goal, row) for goal in priority.goals])


    def _goal_is_satisfied(self, goal: Goal, row: pd.DataFrame):

        # Threshold is float
        if isinstance(goal.thresh, float):
            thresh = goal.thresh

        # Threshold is tag
        elif goal.thresh_func is None:
            thresh = row[goal.thresh].item()

        # Threshold is expression
        else:
            assert goal.thresh_tags is not None # Assertion for pyright
            values = [row[thresh_tag].item() for thresh_tag in goal.thresh_tags]
            thresh = goal.thresh_func(*values)


        x: float = row[goal.tag].to_numpy()[0]
        if goal.op == 'down_to':
            return x <= thresh

        return x >= thresh


    def denormalize_tags(self, df: pd.DataFrame):
        return self._prep_stage.inverse(df)
