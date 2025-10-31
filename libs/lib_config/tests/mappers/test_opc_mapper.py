from lib_config.loader import direct_load_config
from lib_config.mappers.opc_config import opc_mapper
from lib_config.schemas.latest.opc_config import OPCConfig
from lib_config.schemas.v0.opc_config import CoreIOConfig


def test_transform_v0_to_v1():
    """
    Transform v0 CoreIOConfig to v1 OPCConfig and verify contents match expected v1 config
    """
    v0_config = direct_load_config(
        CoreIOConfig,
        config_name="tests/schemas/v0/opc_config/full_config.yaml",
    )
    assert isinstance(v0_config, CoreIOConfig)

    v1_transformed = opc_mapper.get_latest(v0_config)
    assert isinstance(v1_transformed, OPCConfig)
    assert v1_transformed.schema_version.major == 1
    assert v1_transformed.schema_version.minor == 0
    assert v1_transformed.schema_version.patch == 0

    v1_expected = direct_load_config(
        OPCConfig,
        config_name="tests/schemas/latest/opc_config/full_config.yaml",
    )
    assert isinstance(v1_expected, OPCConfig)

    assert len(v1_transformed.connections) == len(v1_expected.connections)
    assert len(v1_transformed.tags) == len(v1_expected.tags)

    for i, (transformed_conn, expected_conn) in enumerate(
        zip(v1_transformed.connections, v1_expected.connections, strict=True),
    ):
        assert transformed_conn.connection_id == expected_conn.connection_id, f"Connection {i} ID mismatch"
        assert transformed_conn.connection_url == expected_conn.connection_url, f"Connection {i} URL mismatch"
        assert transformed_conn.client_timeout == expected_conn.client_timeout, f"Connection {i} timeout mismatch"
        assert transformed_conn.application_uri == expected_conn.application_uri, f"Connection {i} app URI mismatch"

    for i, (transformed_tag, expected_tag) in enumerate(
        zip(v1_transformed.tags, v1_expected.tags, strict=True),
    ):
        assert transformed_tag.name == expected_tag.name, f"Tag {i} name mismatch"
        assert transformed_tag.connection_id == expected_tag.connection_id, f"Tag {i} connection_id mismatch"
        assert transformed_tag.node_identifier == expected_tag.node_identifier, f"Tag {i} node_identifier mismatch"
