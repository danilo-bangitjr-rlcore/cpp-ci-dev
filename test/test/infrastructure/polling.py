import time


def wait_for_agent_state(
    base_url: str,
    agent_id: str,
    expected_state: str,
    max_poll_time: float = 180.0,
    poll_interval: float = 2.0,
    timeout_for_request: float = 5.0,
):
    """
    Poll agent status endpoint until agent reaches expected state or timeout.

    Raises AssertionError if agent fails or timeout expires.
    """
    import requests

    deadline = time.time() + max_poll_time
    last_status = None

    while time.time() < deadline:
        status_response = requests.get(
            f"{base_url}/api/agents/{agent_id}/status",
            timeout=timeout_for_request,
        )
        assert status_response.status_code == 200

        status_data = status_response.json()
        state = status_data.get("state")
        last_status = status_data

        if state == expected_state:
            return status_data

        if state == "failed":
            raise AssertionError(f"Agent failed during execution: {status_data}")

        time.sleep(poll_interval)

    raise AssertionError(
        f"Agent did not reach '{expected_state}' within {max_poll_time}s. Last status: {last_status}",
    )
