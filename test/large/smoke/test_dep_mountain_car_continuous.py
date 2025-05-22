import json
import shutil
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pytest import FixtureRequest
from sqlalchemy import Engine

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.db.data_reader import DataReader, TagDBConfig
from corerl.data_pipeline.pipeline import Pipeline
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from corerl.messages.event_bus import DummyEventBus
from corerl.state import AppState

raw_service_names = [
    # "core-rl-corerl-1",
    "core-rl-telegraf-1",
    "core-rl-renderer-1",
    "core-rl-grafana-1",
    "core-rl-opc-sim-1",
    "core-rl-core-ui-1",
    "core-rl-timescale-db-1",
    "core-rl-opc-server-1",
]

db_cfg = TagDBConfig(
    db_name="postgres",
    table_name="sensors",
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
            "--tag-entries",
            "all",
            "--action-entries",
            "all",
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
def run_coreio(request: FixtureRequest, run_docker_compose: None):
    proc = subprocess.run(
        ["coreio_main", "--config-name", "dep_mountain_car_continuous"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=request.config.rootpath,
    )
    proc.check_returncode()

@pytest.fixture(scope="module")
def check_sim_farama_environment_ready(run_docker_compose: None, request: FixtureRequest):
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
                ["action-0", "tag-0", "tag-1"],
                end - timedelta(seconds=10),
                end
            )

            successful_query = not bool(df.isnull().values.any())
        except Exception:
            # sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable) relation "public.opcua" does not exist
            query_attempts += 1
            time.sleep(1)

        finally:
            if data_reader is not None:
                data_reader.close()


@pytest.mark.skipif(
    should_skip_test(), reason="Docker compose ps saw core-rl services, or failed to run, do not run opc tsdb test"
)
@pytest.mark.timeout(500)
def test_dep_mountain_car_continuous(
    check_sim_farama_environment_ready: None,
    request: FixtureRequest,
    free_localhost_port: int,
):
    event_bus_url = f'tcp://localhost:{free_localhost_port}'
    # coreio_origin = 'tcp://localhost:5558'
    proc = subprocess.run(
        [
            "corerl_main",
            "--config-name",
            "dep_mountain_car_continuous",
            "max_steps=25",
            "interaction.obs_period=00:00:01",
            "interaction.update_period=00:00:01",
            "interaction.action_period=00:00:01",
            "event_bus.cli_connection=" + event_bus_url,
            # "coreio.coreio_origin=" + coreio_origin,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=request.config.rootpath,
    )
    assert proc.returncode == 0, proc.stdout

def test_agent_checkpoint(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    cfg = direct_load_config(
        MainConfig,
        overrides={
            'infra.db.db_name': tsdb_tmp_db_name,
            'infra.db.port': str(tsdb_engine.url.port),
        },
        base=".",
        config_name="config/dep_mountain_car_continuous",
    )
    assert isinstance(cfg, MainConfig)

    event_bus = DummyEventBus()
    app_state = AppState(
        cfg=cfg,
        metrics=MetricsTable(cfg.metrics),
        evals=EvalsTable(cfg.evals),
        event_bus=event_bus,
    )
    pipeline = Pipeline(app_state, cfg.pipeline)
    column_desc = pipeline.column_descriptions
    agent = GreedyAC(
        cfg.agent,
        app_state,
        column_desc,
    )
    now = datetime.now(UTC)
    output_dir = Path('test/large/smoke/test_save_agent/')
    path = output_dir / str(now).replace(':','_')
    path.mkdir(exist_ok=True, parents=True)
    agent.save(path)
    agent.load(path)
    shutil.rmtree(output_dir)


@pytest.mark.skipif(
    should_skip_test(), reason="Docker compose ps saw core-rl services, or failed to run, do not run opc tsdb test"
)
def test_docker_compose_up_running(check_sim_farama_environment_ready: None):
    """This test checks if docker compose up was run successfully by inspecting all
    relevant services and verifying that their container state is running.
    """
    proc = subprocess.run(
        ["docker", "container", "ls", "-a", "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    proc.check_returncode()

    containers_to_check = set(raw_service_names)
    raw_container_outputs = proc.stdout.splitlines()
    for raw_container_output in raw_container_outputs:
        container_output = json.loads(raw_container_output)
        assert "State" in container_output and "Names" in container_output

        state = container_output["State"]
        name = container_output["Names"]

        if name not in raw_service_names:
            continue

        assert state == "running", f"{name} should be running, got {state}"
        containers_to_check.remove(container_output["Names"])

    assert len(containers_to_check) == 0, f"did not find expected container(s): {containers_to_check}"
