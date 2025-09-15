pytest_plugins = (
    # local infra
    'tests.infrastructure.config',
    "test.infrastructure.app_state",
    "test.infrastructure.utils.tsdb",
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
)
