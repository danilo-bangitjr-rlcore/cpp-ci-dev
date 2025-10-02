from datetime import timedelta
from urllib.error import URLError
from urllib.request import urlopen


def check_http_health(
    host: str,
    port: int,
    timeout: timedelta,
    endpoint: str = "/api/healthcheck",
) -> bool:
    """
    Check HTTP service health via endpoint.
    """
    try:
        url = f"http://{host}:{port}{endpoint}"
        with urlopen(url, timeout=timeout.total_seconds()) as response:
            return response.status == 200

    except (URLError, OSError):
        return False
