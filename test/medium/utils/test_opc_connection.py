import pytest
import asyncio
from test.medium.utils.fixture_opc import FakeOpcServer, OpcConnection
from test.medium.utils.fixture_opc import *  # noqa: F403


# Skip if coverage (--cov) option is enabled when calling pytest
@pytest.fixture
def skip_coverage_fixture(request):
    if request.config.getoption("--cov"):
        raise pytest.skip("Test skipped because coverage is emitted")


skip_test_if_coverage = pytest.mark.usefixtures("skip_coverage_fixture")


@skip_test_if_coverage
async def test_connect1(server_and_client: tuple[FakeOpcServer, OpcConnection]):
    """
    Client should be able to connect to a running server.
    """
    _, client = server_and_client
    await client.connect()


@skip_test_if_coverage
async def test_connect2(client: OpcConnection):
    """
    Client should fail when no server is running.
    """
    with pytest.raises(Exception):  # noqa: B017 - OPC does not raise consistent exception
        await client.connect()


@skip_test_if_coverage
async def test_read_values1(server_and_client: tuple[FakeOpcServer, OpcConnection]):
    """
    Client can read values for both sensors.
    """
    server, client = server_and_client

    await client.connect()
    nodes = [
        client.client.get_node('ns=2;i=2'),
        client.client.get_node('ns=2;i=3'),
    ]

    await server.step(1.1)

    values = await client.read_values(nodes)
    assert values == [1.1, 2.1]


@skip_test_if_coverage
async def test_disconnect1(server_and_client: tuple[FakeOpcServer, OpcConnection]):
    """
    Client survives when a server goes offline after connection.
    Check this sequence:
      1. Client and server connect
      2. Server closes
      3. Server starts
      4. Client implicitly reconnects in the background
      5. Client reads
    """
    server, client = server_and_client

    await client.connect()
    nodes = [
        client.client.get_node('ns=2;i=2')
    ]

    await server.step(2.0)
    got = await client.read_values(nodes)
    assert got == [2.0]

    await server.close()
    await asyncio.sleep(0.1)
    await server.start()
    await server.step(3.0)
    got = await client.read_values(nodes)
    assert got == [3.0]


@skip_test_if_coverage
async def test_disconnect2(server_and_client: tuple[FakeOpcServer, OpcConnection]):
    """
    Client survives when a server goes offline after connection.
    Check this sequence:
      1. Client and server connect
      2. Server closes
      3. Client reads
      4. Server starts
      5. Client implicitly reconnects in the background
      6. Client completes read from step 3
    """
    server, client = server_and_client

    await client.connect()
    nodes = [
        client.client.get_node('ns=2;i=2')
    ]

    await server.step(2.0)
    got = await client.read_values(nodes)
    assert got == [2.0]

    await server.close()
    read_future = client.read_values(nodes)
    await asyncio.sleep(0.1)
    await server.start()
    await server.step(3.0)
    got = await read_future
    assert got == [3.0]
