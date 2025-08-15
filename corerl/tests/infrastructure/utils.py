import pytest


def get_fixture[T](
    request: pytest.FixtureRequest,
    fixture_name: str,
    return_type: type[T],
) -> T:
    return request.getfixturevalue(fixture_name)
