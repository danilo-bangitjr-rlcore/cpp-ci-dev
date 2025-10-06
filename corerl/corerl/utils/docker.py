from docker import DockerClient


def container_exists(client: DockerClient, name: str) -> bool:
    containers = client.containers.list(all=True)
    container_names = [c.name for c in containers]
    return name in container_names
