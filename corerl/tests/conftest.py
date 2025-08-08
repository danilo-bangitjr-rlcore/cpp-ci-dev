pytest_plugins = (
    # local infra
    'tests.infrastructure.config',
    'tests.infrastructure.databases',

    # `test/` sub-repo
    "test.infrastructure.app_state",
    "test.infrastructure.networking",
    "test.infrastructure.utils.docker",
    "test.infrastructure.utils.pandas",
    "test.infrastructure.utils.tsdb",
)
