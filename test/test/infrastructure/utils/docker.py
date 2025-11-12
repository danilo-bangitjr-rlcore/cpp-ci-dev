from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from docker import errors, from_env


def init_docker_container(
    name: str,
    repository: str,
    tag: str = "latest",
    env: Mapping[str, str] = MappingProxyType({}),
    ports: Mapping[str, Any] = MappingProxyType({}),
    restart: bool = True,
):
    """Initialize a Docker container with the specified configuration.

    This is a generic Docker container initialization function that does not
    assume any specific database or service type. For database-specific
    containers, use the appropriate wrapper functions.
    """
    client = from_env()

    # Ensure that no container with the same name exists
    running_containers = client.containers.list(filters={"name": name})
    if not restart and len(running_containers) > 0:
        return running_containers[0]

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

    return client.containers.run(
        image,
        detach=True,
        ports=dict(ports),
        environment=dict(env),
        name=name,
    )
