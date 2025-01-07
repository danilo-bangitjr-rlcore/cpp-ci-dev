import subprocess
from os import path, makedirs

import boto3
import pytest
from pytest import FixtureRequest
from docker.models.containers import Container
from botocore.exceptions import NoCredentialsError
from sqlalchemy import URL

from corerl.sql_logging.sql_logging import try_create_engine
from corerl.data_pipeline.db.utils import try_connect
from test.infrastructure.utils.docker import init_docker_container

def should_skip_test():
    try:
        boto3.Session().get_credentials()
    except NoCredentialsError:
        # no credentials, skip test
        return True

    sts = boto3.client('sts')
    try:
        sts.get_caller_identity()
    except Exception:
        # credentials cannot get identity, skip test
        return True


    s3 = boto3.client('s3')
    try:
        # will throw if not found or no permission to access bucket
        s3.head_object(
            Bucket="rlcore-shared",
            Key="epcor/Epcor_Scrubber-DB_Exports/partial/scrubber4_2024_12_15.sql"
        )
    except Exception:
        # credentials cannot access s3 rlcore-shared bucket, skip test
        return True

    return False


@pytest.fixture(scope="module")
def init_epcor_tsdb_container():
    container = init_docker_container(
        name="epcor_tsdb_scrubber",
        ports={"5432": 5434}
    )
    yield container
    container.stop()
    container.remove()

@pytest.fixture(scope="module")
def dl_epcor_partial_data_if_not_exists(init_epcor_tsdb_container: Container, request: FixtureRequest):
    repo_root = request.config.rootpath
    offline_data_dir = path.join(repo_root, "offline_data")

    # make directory if directory does not exist
    if not path.exists(offline_data_dir):
        makedirs(offline_data_dir)

    # download the scrubber data if not already downloaded
    scrubber_raw_sql_dump_file = path.join(offline_data_dir, "scrubber4_2024_12_15.sql")
    if not path.exists(scrubber_raw_sql_dump_file):
        s3 = boto3.client('s3')
        with open(scrubber_raw_sql_dump_file, "wb") as f:
            s3.download_fileobj(
                "rlcore-shared",
                "epcor/Epcor_Scrubber-DB_Exports/partial/scrubber4_2024_12_15.sql",
                f
            )

    # load the raw scrubber data into postgres
    engine = try_create_engine(URL.create(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        host="localhost",
        port=5434,
        database="postgres",
    ))
    # only used to determine that engine is safe to load data into
    conn = try_connect(engine)
    conn.close()

    with open(scrubber_raw_sql_dump_file, "r") as f:
        # very cheeky way of loading in sql dump
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "epcor_tsdb_scrubber",
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres"
            ],
            input=f.read(),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        assert result.check_returncode() is None


@pytest.mark.skipif(
    should_skip_test(),
    reason="Skipping test because boto3 credentials invalid and stub file doesn't exist"
)
def test_epcor_tsdb_scrubber(
    request: FixtureRequest,
    dl_epcor_partial_data_if_not_exists: None,
): # noqa: F811
    """This test loads the partial scrubber4 data into a TSDB instance
    and ensures that we can run the TSDBAsyncStubEnv without any error.

    This verifies:
    * configuration specification of our database, tags, experiment
    * communication between our datareader and the pipeline

    This test does not verify that we have an agent that is learning a policy.
    """
    repo_root = request.config.rootpath
    result = subprocess.run(
        [
            "corerl_main",
            "--config-name",
            "epcor_tsdb_scrubber",
            "experiment.max_steps=25",
            "pipeline.db.port=5434"
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )
    assert result.check_returncode() is None
