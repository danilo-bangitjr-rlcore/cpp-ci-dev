"""Unit tests for metric filtering logic."""

from coretelemetry.agent_metrics_api.metric_filters import (
    METRIC_FILTERS,
    MetricFilter,
    filter_metrics,
    get_metric_description,
)


class TestMetricFilter:
    """Tests for MetricFilter dataclass."""

    def test_matches_case_insensitive(self):
        """Test pattern matching is case insensitive."""
        filter_config = MetricFilter(pattern=r'^q$', description='Quality')

        assert filter_config.matches('q')
        assert filter_config.matches('Q')

    def test_matches_complex_pattern(self):
        """Test complex regex pattern matching."""
        filter_config = MetricFilter(
            pattern=r'^pipeline_[a-z0-9_]+_mean$',
            description='Pipeline mean',
        )

        assert filter_config.matches('pipeline_foo_bar_baz_mean')
        assert filter_config.matches('pipeline_123_abc_mean')
        assert not filter_config.matches('pipeline_mean')
        assert not filter_config.matches('pipeline_foo_bar_variance')


class TestGetMetricDescription:
    """Tests for get_metric_description function."""

    def test_returns_description_for_matching_metric(self):
        """Test description is returned for metrics matching a filter."""
        description = get_metric_description('q')
        assert description == 'Quality (value) of the action'

    def test_returns_none_for_non_matching_metric(self):
        """Test None is returned for metrics not matching any filter."""
        description = get_metric_description('unknown_metric_xyz')
        assert description is None

    def test_case_insensitive_matching(self):
        """Test matching works regardless of case."""
        description = get_metric_description('Q')
        assert description == 'Quality (value) of the action'

    def test_first_match_wins(self):
        """Test that first matching filter's description is returned."""
        # 'q' should match the first '^q$' pattern
        description = get_metric_description('q')
        assert description is not None


class TestFilterMetrics:
    """Tests for filter_metrics function."""

    def test_filters_matching_metrics(self):
        """Test only metrics matching patterns are returned."""
        metrics = ['q', 'temperature', 'pressure', 'critic0_loss']
        result = filter_metrics(metrics)

        assert len(result) == 2
        assert result[0]['name'] == 'q'
        assert result[0]['description'] == 'Quality (value) of the action'
        assert result[1]['name'] == 'critic0_loss'
        assert 'well the agent understands' in result[1]['description']

    def test_empty_list_returns_empty(self):
        """Test empty input returns empty output."""
        result = filter_metrics([])
        assert result == []

    def test_no_matches_returns_empty(self):
        """Test no matches returns empty list."""
        metrics = ['temperature', 'pressure', 'humidity']
        result = filter_metrics(metrics)
        assert result == []

    def test_all_matches_returns_all(self):
        """Test all matching metrics are returned with descriptions."""
        metrics = ['q', 'q_ensemble_variance', 'ae-imputed']
        result = filter_metrics(metrics)

        assert len(result) == 3
        assert all('name' in item and 'description' in item for item in result)

    def test_pipeline_metrics_filtering(self):
        """Test pipeline metrics are correctly filtered."""
        metrics = [
            'pipeline_init_foo_bar_mean',
            'pipeline_init_foo_bar_variance',
            'pipeline_init_foo_bar_max',
            'pipeline_init_foo_bar_min',
            'other_metric',
        ]
        result = filter_metrics(metrics)

        assert len(result) == 4
        assert all('pipeline' in item['name'] for item in result)
        assert all('statistics about the tags' in item['description'] for item in result)

    def test_ae_metrics_filtering(self):
        """Test ae- prefix metrics are filtered correctly."""
        metrics = [
            'ae-num_nan_obs',
            'ae-imputed',
            'ae-quality-foo_trace_bar',
            'temperature',
        ]
        result = filter_metrics(metrics)

        assert len(result) == 3
        assert result[0]['name'] == 'ae-num_nan_obs'
        assert result[1]['name'] == 'ae-imputed'
        assert result[2]['name'] == 'ae-quality-foo_trace_bar'


class TestMetricFiltersConfiguration:
    """Tests for METRIC_FILTERS configuration."""

    def test_all_filters_have_pattern_and_description(self):
        """Test all filters have required fields."""
        for filter_config in METRIC_FILTERS:
            assert filter_config.pattern
            assert filter_config.description
            assert isinstance(filter_config.pattern, str)
            assert isinstance(filter_config.description, str)

    def test_filters_cover_expected_categories(self):
        """Test filters cover main metric categories."""
        patterns = [f.pattern for f in METRIC_FILTERS]

        # Check key patterns exist
        assert any('q$' in p for p in patterns)
        assert any('critic' in p for p in patterns)
        assert any('pipeline' in p for p in patterns)
        assert any('ae-' in p for p in patterns)
        assert any('observed_a_q' in p for p in patterns)
        assert any('partial_return' in p for p in patterns)
