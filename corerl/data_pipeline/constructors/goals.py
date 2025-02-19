import numba
import numpy as np
import pandas as pd

from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.datatypes import PipelineFrame
from corerl.data_pipeline.tag_config import TagConfig, get_tag_bounds
from corerl.environment.reward.config import Goal, JointGoal, Optimization, RewardConfig
from corerl.utils.maybe import Maybe


class GoalConstructor:
    def __init__(self, reward_cfg: RewardConfig, tag_cfgs: list[TagConfig], prep_stage: Preprocessor):
        self._cfg = reward_cfg
        self._tag_cfgs = tag_cfgs
        self._prep_stage = prep_stage

    def __call__(self, pf: PipelineFrame) -> PipelineFrame:
        # denormalize all tags before checking constraint violations
        pf.rewards = self.denormalize_tags(pf.data)

        rewards = []
        for (_, row_series) in pf.rewards.iterrows():
            row = row_series.to_frame().transpose()
            active_idx = self._find_active_priority_idx(row)
            priority = self._cfg.priorities[active_idx]
            if not isinstance(priority, Optimization):
                violation_percent = self._priority_violoation_percent(priority, row)

                # break [-1, -0.5] into num_priorities-1 buckets
                num_buckets = len(self._cfg.priorities) - 1
                buckets = np.linspace(-1, -0.5, num_buckets + 1)
                r = _put_in_range(-violation_percent, old_range=(-1, 0), new_range=buckets[active_idx:active_idx+1])

            else:
                opt = self._avg_optimization_violation(priority, row)
                r = _put_in_range(opt, old_range=(-1, 0), new_range=(-0.5, 0))

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
            .map(get_tag_bounds)
            .expect(f'Was unable to find tag config for tag: {tag}')
        )
        lo = bounds[0].expect(f'Was unable to find a lower bound for tag: {tag}')
        hi = bounds[1].expect(f'Was unable to find an upper bound for tag: {tag}')

        x: float = row[tag].to_numpy()[0]
        if dir == 'max':
            return (hi - x) / (hi - lo)

        return (x - lo) / (hi - lo)


    def _priority_violoation_percent(self, priority: Goal | JointGoal, row: pd.DataFrame) -> float:
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
        """
        if isinstance(priority, JointGoal):
            violation_percents = [self._priority_violoation_percent(goal, row) for goal in priority.goals]
            if priority.op == 'and':
                return np.max(violation_percents)
            return np.min(violation_percents)

        return self._goal_violation_percent(priority, row)


    def _goal_violation_percent(self, goal: Goal, row: pd.DataFrame):
        """
        To produce a sloped reward for violating a goal, we need the degree of violation.
        This degree of violation is expressed as a percent of the range of the tag.

        If we are trying to decrease a tag, then the percent of violation is from
        the tag's upper bound to the target threshold. If we are increasing a tag,
        then the percent violoation from tag's lower bound to the target threshold.
        """
        bounds = (
            Maybe.find(lambda cfg: cfg.name == goal.tag, self._tag_cfgs)
            .map(get_tag_bounds)
            .expect(f'Was unable to find tag config for tag: {goal.tag}')
        )

        x: float = row[goal.tag].to_numpy()[0]
        if goal.op == 'down_to':
            hi = bounds[1].expect(f'Was unable to find an upper bound for tag: {goal.tag}')
            delta = x - goal.thresh
            return delta / (hi - goal.thresh)

        lo = bounds[0].expect(f'Was unable to find a lower bound for tag: {goal.tag}')
        delta = goal.thresh - x
        return delta / (goal.thresh - lo)


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
        x: float = row[goal.tag].to_numpy()[0]
        if goal.op == 'down_to':
            return x <= goal.thresh

        return x >= goal.thresh


    def denormalize_tags(self, df: pd.DataFrame):
        return self._prep_stage.inverse(df)


@numba.njit
def _put_in_range(
    x: np.ndarray | float,
    old_range: tuple[float, float] | np.ndarray,
    new_range: tuple[float, float] | np.ndarray,
):
    """
    Take a float value x in old_range and map it to a value in new_range.
    E.g. if x is 0.5 in [0, 1] and new_range is [-1, 1], then return 0.
    """
    old_d = (old_range[1] - old_range[0])
    new_d = (new_range[1] - new_range[0])
    return (((x - old_range[0]) * new_d) / old_d) + new_range[0]
