"""Metric filtering and classification for agent metrics."""

import re
from dataclasses import dataclass


@dataclass
class MetricFilter:
    """A filter pattern with associated description for metric classification.

    Attributes:
        pattern: Regular expression pattern (case-insensitive)
        description: Human-readable description of what this metric represents
    """
    pattern: str
    description: str

    def matches(self, metric_name: str) -> bool:
        """Check if metric name matches this filter pattern.

        Args:
            metric_name: The metric name to test

        Returns:
            True if metric matches pattern, False otherwise
        """
        return bool(re.match(self.pattern, metric_name, re.IGNORECASE))


# Standard metric filters with patterns and descriptions
METRIC_FILTERS = [
    MetricFilter(
        pattern=r'^q$',
        description='Quality (value) of the action',
    ),
    MetricFilter(
        pattern=r'^q_ensemble_variance$',
        description="The agent's uncertainty about the action it chose",
    ),
    MetricFilter(
        pattern=r'^observed_a_q[a-z0-9_]*$',
        description=(
            "The difference between observed_a_q{label} and partial_return{label} "
            "can be interpreted as the agent's prediction accuracy"
        ),
    ),
    MetricFilter(
        pattern=r'^partial_return[a-z0-9_]*$',
        description=(
            "The difference between observed_a_q{label} and partial_return{label} "
            "can be interpreted as the agent's prediction accuracy"
        ),
    ),
    MetricFilter(
        pattern=r'^pdf_plot_action_[a-z0-9_]+$',
        description='Probability of choosing each unrealized alternative',
    ),
    MetricFilter(
        pattern=r'^qs_plot_action_[a-z0-9_]+$',
        description='The value of unrealized alternatives',
    ),
    MetricFilter(
        pattern=r'^critic[0-9]*_loss$',
        description='How well the agent understands the data seen so far. Less is better (think of it like error)',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_num_non_nan$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_num_nan$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_mean$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_variance$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_max$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_min$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_50th_percentile$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_90th_percentile$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_percent_nan$',
        description='They tell us statistics about the tags flowing into the agent',
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_average_length_of_nan_chunks$',
        description=(
            'These could all be exposed for the INIT stage of the pipeline. '
            'They tell us statistics about the tags flowing into the agent'
        ),
    ),
    MetricFilter(
        pattern=r'^pipeline_[a-z0-9_]+_[a-z0-9_]+_[a-z0-9_]+_average_length_of_non_nan_chunks$',
        description=(
            'These could all be exposed for the INIT stage of the pipeline. '
            'They tell us statistics about the tags flowing into the agent'
        ),
    ),
    MetricFilter(
        pattern=r'^ae-num_nan_obs$',
        description='Number of missing values in observations',
    ),
    MetricFilter(
        pattern=r'^ae-imputed$',
        description='Number of values imputed',
    ),
    MetricFilter(
        pattern=r'^ae-quality-[a-z0-9_]+_trace_[a-z0-9_]+$',
        description=(
            'Quality of state variables input to agent. '
            'Can be made more interpretable by averaging across decays. '
            'This gives a quality per tag'
        ),
    ),
]


def get_metric_description(metric_name: str) -> str | None:
    """Get description for a metric if it matches any filter pattern.

    Args:
        metric_name: The metric name to look up

    Returns:
        Description string if metric matches a filter, None otherwise
    """
    for filter_config in METRIC_FILTERS:
        if filter_config.matches(metric_name):
            return filter_config.description
    return None


def filter_metrics(metric_names: list[str]) -> list[dict[str, str]]:
    """Filter metrics and attach descriptions based on patterns.

    Args:
        metric_names: List of metric names to filter

    Returns:
        List of dicts with 'name' and 'description' keys for metrics that match filters
    """
    filtered = []
    for metric_name in metric_names:
        description = get_metric_description(metric_name)
        if description is not None:
            filtered.append({
                "name": metric_name,
                "description": description,
            })
    return filtered
