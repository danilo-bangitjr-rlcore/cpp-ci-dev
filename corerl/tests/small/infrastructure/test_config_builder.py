import pytest
from lib_config.loader import direct_load_config

from corerl.config import MainConfig
from tests.infrastructure.config import ConfigBuilder


class TestConfigBuilder:

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        config = direct_load_config(MainConfig, config_name='tests/infrastructure/configs/basic_config.yaml')
        assert isinstance(config, MainConfig)
        return config

    def test_override_functionality(self, sample_config: MainConfig):
        """
        Validates core override capabilities including single overrides, method chaining,
        and bulk dictionary overrides. Ensures all override patterns work correctly.
        """
        builder = ConfigBuilder(sample_config)
        original_gamma = sample_config.agent.gamma
        original_seed = sample_config.seed

        # Test single override
        config1 = builder.with_override('agent.gamma', 0.95).build()
        assert config1.agent.gamma == 0.95

        # Test method chaining
        config2 = (
            builder
            .with_override('agent.gamma', 0.99)
            .with_override('seed', 42)
            .with_override('max_steps', 1000)
            .build()
        )

        assert config2.agent.gamma == 0.99
        assert config2.seed == 42
        assert config2.max_steps == 1000

        # Test bulk dictionary overrides
        overrides = {
            'agent.gamma': 0.85,
            'seed': 123,
            'max_steps': 500,
        }
        config3 = ConfigBuilder(sample_config).with_overrides(overrides).build()

        assert config3.agent.gamma == 0.85
        assert config3.seed == 123
        assert config3.max_steps == 500

        # Verify original config unchanged
        assert sample_config.agent.gamma == original_gamma
        assert sample_config.seed == original_seed

    def test_nested_attributes(self, sample_config: MainConfig):
        """
        Validates handling of deeply nested attribute paths and complex object navigation.
        Tests the dot-notation path parsing for multi-level configuration structures.
        """
        builder = ConfigBuilder(sample_config)

        config = (
            builder
            .with_override('agent.critic.stepsize', 0.005)
            .with_override('interaction.obs_period', '00:00:30')
            .build()
        )

        assert config.agent.critic.stepsize == 0.005
        assert str(config.interaction.obs_period) == '00:00:30'

    def test_error_handling(self, sample_config: MainConfig):
        """
        Validates proper error handling for invalid attribute paths.
        Ensures meaningful error messages for debugging configuration issues.
        """
        builder = ConfigBuilder(sample_config)

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            builder.with_override('agent.nonexistent.gamma', 0.95).build()

        with pytest.raises(AttributeError, match="Object has no attribute"):
            builder.with_override('invalid', 42).build()

    def test_immutability(self, sample_config: MainConfig):
        """
        Validates that ConfigBuilder operations never modify the original configuration object.
        Critical for test isolation and preventing side effects between test runs.
        """
        original_gamma = sample_config.agent.gamma
        original_seed = sample_config.seed

        builder = ConfigBuilder(sample_config)

        # Build multiple configs with different overrides
        config1 = builder.with_override('agent.gamma', 0.95).build()
        config2 = builder.with_override('seed', 42).build()

        # Original config must remain unchanged
        assert sample_config.agent.gamma == original_gamma
        assert sample_config.seed == original_seed

        # Built configs should have expected independent values
        assert config1.agent.gamma == 0.95
        assert config2.agent.gamma == 0.95  # Inherits from builder state
        assert config2.seed == 42
