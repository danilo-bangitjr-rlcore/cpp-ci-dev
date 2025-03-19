import filelock
import pytest

from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine
from test.behavior.bsuite import BSuiteTestCase
from test.behavior.distraction_world.cases import DistractionWorldTest
from test.behavior.mountain_car.cases import MountainCar, StandStillMountainCar
from test.behavior.saturation.cases import (
    DelayedSaturationTest,
    DeltaSaturationTest,
    SaturationTest,
)
from test.behavior.windy_room.cases import WindyRoomTest
from test.infrastructure.utils.docker import init_docker_container


@pytest.fixture(scope='session')
def bsuite_tsdb():
    # file locking is necessary because we want to run these tests
    # in parallel using xdist, and we can't have a true session scope.
    # this prevents every test from independently trying to setup docker

    # Note: we aren't cleaning up after ourselves here, because the bsuite
    # tsdb docker container is intended to be long-lived
    with filelock.FileLock('output/bsuite.lock'):
        init_docker_container(
            name='tsdb-bsuite',
            restart=False,
            ports={'5432': 22222},
        )



TEST_CASES = [
    DelayedSaturationTest(),
    DeltaSaturationTest(),
    MountainCar(),
    SaturationTest(),
    StandStillMountainCar(),
    DistractionWorldTest(),
    WindyRoomTest(),
]

@pytest.mark.parametrize('test_case', TEST_CASES)
@pytest.mark.timeout(900)
def test_bsuite(
    test_case: BSuiteTestCase,
    bsuite_tsdb: None,
):
    PORT = 22222
    db_name = 'bsuite'
    schema = test_case.name.lower().replace(' ', '_')
    cfg = SQLEngineConfig(
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='workstation',
        port=PORT,
    )
    engine = get_sql_engine(cfg, db_name)
    metrics_table = test_case.execute_test(engine, db_name, schema)
    test_case.evaluate_outcomes(engine, metrics_table)
