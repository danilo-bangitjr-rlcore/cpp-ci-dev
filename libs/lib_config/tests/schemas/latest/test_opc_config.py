from pathlib import Path

import pytest

from lib_config.errors import ConfigValidationErrors
from lib_config.loader import direct_load_config
from lib_config.schemas.latest.opc_config import OPCConfig

TEST_DIR = Path(__file__).parent / "opc_config"
BASE_CONFIG = str(TEST_DIR / "base.yaml")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"schema_version": "1.2.3"},
        {"connections[0].client_timeout": "60"},
        {"connections[0].application_uri": "urn:test:app"},
    ],
    ids=[
        "base_config",
        "custom_schema_version",
        "custom_timeout",
        "with_application_uri",
    ],
)
def test_valid_configs(overrides: dict[str, str]):
    """
    Load valid configs using base config with various overrides
    """
    config = direct_load_config(
        OPCConfig,
        config_name=BASE_CONFIG,
        overrides=overrides,
    )
    assert isinstance(config, OPCConfig)


def test_full_config():
    """
    Load complete config with all features enabled
    """
    config = direct_load_config(
        OPCConfig,
        config_name=str(TEST_DIR / "full_config.yaml"),
    )
    assert isinstance(config, OPCConfig)


@pytest.mark.parametrize(
    "yaml_file,expected_error_key",
    [
        ("missing_connection_id.yaml", "connections.connection_id"),
        ("missing_security_policy.yaml", "connections.security_policy"),
        ("missing_server_cert.yaml", "server_cert_path"),
        ("invalid_timeout_type.yaml", "client_timeout"),
        ("invalid_security_policy.yaml", "security_policy"),
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
        OPCConfig,
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
