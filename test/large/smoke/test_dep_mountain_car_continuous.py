import multiprocessing
import subprocess
import time
from datetime import UTC, datetime, timedelta

import pytest
from pytest import FixtureRequest

from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig

raw_service_names = [
    "core-rl-opc-server-1",
    "core-rl-telegraf-1",
    "core-rl-timescale-db-1",
]

db_cfg = TagDBConfig(
    db_name="postgres",
    table_name="opcua",
    table_schema="public",
    drivername="postgresql+psycopg2",
    username="postgres",
    password="password",
    ip="localhost",
    port=5432,
)

def should_skip_test():
    """Skip running large OPC TSDB mountain car smoke test if the
    docker compose services are already up and running.

    Returns `true` if we are skipping test, `false` otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "ps"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            # skip test, running docker compose did not exit cleanly, rest of test will fail
            return True

        raw_stdout = result.stdout.lower()
        for raw_service_name in raw_service_names:
            if raw_service_name in raw_stdout:
                return True  # skip test, a docker compose service was returned in ps

        return False  # docker compose ps ran successfully, no docker compose service name found, run test
    except Exception:
        # docker compose ps threw some exception, skip test
        return True


@pytest.fixture(scope="module")
def run_make_configs(request: FixtureRequest):
    """Generate the configurations for opc_mountain_car_continuous.
    Emitted `generated_telegraf.conf` file needed for docker compose up telegraf service
    """
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "e2e/make_configs.py",
            "--name",
            "MountainCarContinuous-v0",
            "--telegraf",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=request.config.rootpath,
    )
    proc.check_returncode()


@pytest.fixture(scope="module")
def run_docker_compose(run_make_configs: None, request: FixtureRequest):
    """Run docker compose up to spin up services, cleanup after yield
    """
    proc = subprocess.run(
        ["docker", "compose", "up", "-d"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=request.config.rootpath,
    )

    proc.check_returncode()
    yield

    proc = subprocess.run(
        ["docker", "compose", "down"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=request.config.rootpath,
    )
    proc.check_returncode()


@pytest.fixture(scope="module")
def check_opc_server_ready(run_docker_compose: None, request: FixtureRequest):
    """Check for the 'counter' stub value that is emitted within the OPC server,
    ensure that it's being picked up by telegraf.
    """
    start = datetime.now(UTC)
    successful_query = False
    query_attempts = 0
    while not successful_query:
        if query_attempts >= 100:
            raise RuntimeError("Failed to query for Counter in tsdb, are docker compose services failing?")
        data_reader = None
        try:
            end = datetime.now(UTC)
            data_reader = DataReader(db_cfg=db_cfg)

            _df = data_reader.single_aggregated_read(["Counter"], start, end)
            successful_query = True
        except Exception:
            # sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable) relation "public.opcua" does not exist
            query_attempts += 1
            time.sleep(1)
            pass
        finally:
            if data_reader is not None:
                data_reader.close()


@pytest.fixture(scope="module")
def run_background_opc_client(check_opc_server_ready: None, request: FixtureRequest):
    """Run the OPC Client simulated farama environment"""
    # for some reason Popen failes to keep the simulation farama gym running as daemon
    # using proc = subprocess.Popen, use subprocess.run within multiprocessing process instead
    def subproc_run():
        subprocess.run(
            ["uv", "run", "python", "e2e/opc_clients/opc_client.py", "--config-name", "mountain_car_continuous"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=request.config.rootpath,
        )

    process = multiprocessing.Process(target=subproc_run)
    process.start()
    yield

    process.terminate()


@pytest.fixture(scope="module")
def check_sim_farama_environment_ready(run_background_opc_client: None, request: FixtureRequest):
    """Check for the opc_client values that are emitted within the farama gym environment simulation,
    ensure that it's being picked up by telegraf.
    """
    successful_query = False
    query_attempts = 0
    while not successful_query:
        if query_attempts >= 100:
            raise RuntimeError("Failed to query for observations in tsdb, is farama sim failing?")
        data_reader = None
        try:
            end = datetime.now(UTC)
            data_reader = DataReader(db_cfg=db_cfg)

            df = data_reader.single_aggregated_read(
                ["observation_0", "observation_1", "action_0", "gym_reward"],
                end - timedelta(seconds=10),
                end
            )

            print(df)
            successful_query = not bool(df.isnull().values.any())
        except Exception:
            # sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable) relation "public.opcua" does not exist
            query_attempts += 1
            time.sleep(1)
            pass
        finally:
            if data_reader is not None:
                data_reader.close()


@pytest.mark.skipif(
    should_skip_test(), reason="Docker compose ps saw core-rl services, or failed to run, do not run opc tsdb test"
)
@pytest.mark.timeout(500)
def test_dep_mountain_car_continuous(check_sim_farama_environment_ready: None, request: FixtureRequest):
    proc = subprocess.run(
        [
            "corerl_main",
            "--config-name",
            "dep_mountain_car_continuous",
            "experiment.max_steps=25",
            "experiment.run_forever="
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=request.config.rootpath,
    )
    proc.check_returncode()
