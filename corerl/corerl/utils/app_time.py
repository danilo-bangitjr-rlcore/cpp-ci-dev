from datetime import UTC, datetime, timedelta


class AppTime:
    """
    Manages application time and step counter for demo mode and normal operation.
    For demo mode: Time advances deterministically based on agent_step.
    For normal operation: Time reflects real wall-clock time.
    """

    def __init__(
        self,
        is_demo: bool,
        start_time: datetime,
        obs_period: timedelta | None = None,
        agent_step: int = 0,
    ):
        self.is_demo = is_demo
        self.start_time = start_time

        if self.is_demo:
            if obs_period is None:
                raise ValueError("Must include obs_period for demo mode")

        self.obs_period = obs_period
        self.agent_step = agent_step

    def get_current_time(self) -> datetime:
        """Get current application time."""
        if self.is_demo:
            assert self.obs_period is not None
            return self.start_time + self.agent_step * self.obs_period
        return datetime.now(UTC)

    def increment_step(self) -> None:
        """Advance to next step (increments agent_step by 1)."""
        self.agent_step += 1

    def __getstate__(self):
        """Support for checkpointing."""
        return {
            'agent_step': self.agent_step,
            'start_time': self.start_time,
            'is_demo': self.is_demo,
            'obs_period': self.obs_period,
        }

    def __setstate__(self, state: dict) -> None:
        """Support for checkpoint restoration."""
        self.agent_step = state['agent_step']
        self.start_time = state['start_time']
        self.is_demo = state['is_demo']
        self.obs_period = state['obs_period']
