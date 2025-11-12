pytest_plugins = (
    "tests.infrastructure.mock_opc_certs",
    "tests.infrastructure.mock_opc_server",
    "test.infrastructure.networking", # note, this is from the `test/` subrepo
    "test.infrastructure.utils.tsdb",
)
