from pathlib import Path

import pytest

from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from lib_config.schemas.v0.opc_config import CoreIOConfig

TEST_DIR = Path(__file__).parent / "opc_config"
BASE_CONFIG = str(TEST_DIR / "base.yaml")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"log_level": "DEBUG"},
        {"log_file": "/var/log/test.log"},
        {"coreio_origin": "tcp://192.168.1.1:5557"},
        {"data_ingress.enabled": "true"},
    ],
    ids=[
        "base_config",
        "custom_log_level",
        "with_log_file",
        "custom_origin",
        "enable_data_ingress",
    ],
)
def test_valid_configs(overrides: dict[str, str]):
    """
    Load valid configs using base config with various overrides
    """
    config = direct_load_config(
        CoreIOConfig,
        config_name=BASE_CONFIG,
        overrides=overrides,
    )
    assert isinstance(config, CoreIOConfig)


def test_full_config():
    """
    Load complete config with all features enabled
    """
    config = direct_load_config(
        CoreIOConfig,
        config_name=str(TEST_DIR / "full_config.yaml"),
    )
    assert isinstance(config, CoreIOConfig)


@pytest.mark.parametrize(
    "yaml_file,expected_error_key",
    [
        ("missing_connection_id.yaml", "opc_connections.connection_id"),
        ("missing_security_policy.yaml", "opc_connections.security_policy"),
        ("missing_server_cert.yaml", ["server_cert_path", "authentication_mode"]),
        ("invalid_timeout_type.yaml", "client_timeout"),
        ("invalid_security_policy.yaml", "policy"),
    ],
    ids=[
        "missing_connection_id",
        "missing_security_policy",
        "missing_server_cert",
        "invalid_timeout_type",
        "invalid_security_policy",
    ],
)
def test_validation_errors(yaml_file: str, expected_error_key: str | list[str]):
    """
    Validation errors surface correctly for invalid configs
    """
    result = direct_load_config(
        CoreIOConfig,
        config_name=str(TEST_DIR / yaml_file),
    )
    assert isinstance(result, ConfigValidationErrors)

    if isinstance(expected_error_key, list):
        assert any(
            any(key_part in error_key for key_part in expected_error_key)
            for error_key in result.meta.keys()
        )
    else:
        assert any(expected_error_key in key for key in result.meta.keys())
