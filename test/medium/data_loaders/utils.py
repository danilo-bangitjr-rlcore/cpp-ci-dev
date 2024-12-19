import pytest
from corerl.utils.docker import container_exists, stop_container
from docker import DockerClient, from_env, errors

PERSIST = False  # if true, stop container but don't remove it (data will persist)
INSPECT = False  # if true, leave container running after tests conclude

def create_timescale_container(client: DockerClient, name: str) -> None:
    if container_exists(client, name):
        return

    try:
        image = client.images.get("timescale/timescaledb-ha:pg16")
    except errors.ImageNotFound:
        image = client.images.pull("timescale/timescaledb-ha", "pg16")

    env = {"POSTGRES_PASSWORD": "password"}
    client.containers.create(
        image=image,
        detach=True,
        ports={"5432": 5433},
        environment=env,
        name=name,
    )

def start_timescale_container(client: DockerClient, name: str):
    create_timescale_container(client, name)
    container = client.containers.get(name)
    container.start()


@pytest.fixture(scope="session", autouse=True)
def timescale_docker():
    client = from_env()
    container_name = "test_timescale"
    start_timescale_container(client, name=container_name)
    yield

    # code after the yield is executed at cleanup time
    if INSPECT:
        return

    stop_container(client, name=container_name)

    if PERSIST:
        return

    # prune container, deleting test data
    client.containers.prune()
