"""Basic tests for CoreTelemetry."""


def test_basic_sanity() -> None:
    """
    This is a placeholder test to ensure the test infrastructure is working.
    """
    assert True


def test_with_fixture(sample_fixture: dict[str, str]) -> None:
    """
    This demonstrates the fixture system is working correctly.
    """
    assert sample_fixture["status"] == "ok"
