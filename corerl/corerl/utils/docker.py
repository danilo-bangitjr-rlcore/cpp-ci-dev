from docker import DockerClient


def container_exists(client: DockerClient, name: str) -> bool:
    containers = client.containers.list(all=True)
    container_names = [c.name for c in containers]
    exists = name in container_names
    return exists

def stop_container(client: DockerClient, name: str):
    if not container_exists(client, name):
        return
    container = client.containers.get(name)
    container.stop()
