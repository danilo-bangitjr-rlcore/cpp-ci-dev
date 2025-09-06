import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from docker import errors, from_env
from docker.models.containers import Container


def wait_for_postgres_ready(container: Container, timeout: int = 60, check_interval: float = 1.0):
    """
    Wait for PostgreSQL container to be ready to accept connections.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Check if container is running
            container.reload()
            if container.status != 'running':
                time.sleep(check_interval)
                continue

            # Check PostgreSQL logs for ready signal
            logs = container.logs(tail=50).decode('utf-8')

            # Look for PostgreSQL ready indicators in logs
            ready_indicators = [
                'database system is ready to accept connections',
                'PostgreSQL init process complete',
                'listening on IPv4 address',
            ]

            if any(indicator in logs for indicator in ready_indicators):
                # Double-check with a simple connection test via container exec
                result = container.exec_run('pg_isready -h localhost -p 5432 -U postgres')
                if result.exit_code == 0:
                    return True

        except Exception:
            pass

        time.sleep(check_interval)

    container_id = container.id[:12] if container.id else "unknown"
    raise TimeoutError(f"PostgreSQL container {container_id} not ready after {timeout} seconds")


def init_docker_container(
    name: str = "test_timescale",
    repository: str = "timescale/timescaledb",
    tag: str = "latest-pg17",
    env: Mapping[str, str] = MappingProxyType({"POSTGRES_PASSWORD": "password"}),
    ports: Mapping[str, Any] = MappingProxyType({"5432": 5433}),
    restart: bool = True,
):
    """This function will spin up a docker container running timescaledb.
    All docker container and image configurations are parameterized.
    By default, a timescaledb:latest-pg17 image will be used to create a container named
    test_timescale, binding container port 5432 to 5433.
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

    container = client.containers.run(
        image,
        detach=True,
        ports=dict(ports),
        environment=dict(env),
        name=name,
    )

    # Wait for PostgreSQL to be ready before returning
    try:
        wait_for_postgres_ready(container, timeout=60)
    except TimeoutError as e:
        # Clean up container if it fails to start properly
        container.stop()
        container.remove()
        raise e

    return container
