from types import MappingProxyType
from typing import Any, Mapping

from docker import errors, from_env


def init_docker_container(
    name: str = "test_timescale",
    repository: str = "timescale/timescaledb-ha",
    tag: str = "pg16",
    env: Mapping[str, str] = MappingProxyType({"POSTGRES_PASSWORD": "password"}),
    ports: Mapping[str, Any] = MappingProxyType({"5432": 5433})
):
    """This function will spin up a docker container running timescaledb.
    All docker container and image configurations are parameterized.
    By default, a timescaledb-ha:pg16 image will be used to create a container named
    test_timescale, binding container port 5432 to 5433.
    """
    client = from_env()

    # Ensure that no container with the same name exists
    try:
        existing_container = client.containers.get(name)
        existing_container.stop()
        existing_container.remove()
    except errors.NotFound:
        pass

    # Ensure that the requested image exists
    try:
        image = client.images.get(f"{repository}:{tag}")
    except errors.ImageNotFound:
        image = client.images.pull(repository, tag)

    container = client.containers.run(
        image,
        detach=True,
        ports=dict(ports),
        environment=dict(env),
        name=name,
    )

    return container
