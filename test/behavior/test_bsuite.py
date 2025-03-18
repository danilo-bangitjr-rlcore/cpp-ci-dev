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

TEST_CASES = [
    DelayedSaturationTest(),
    DeltaSaturationTest(),
    MountainCar(),
    SaturationTest(),
    StandStillMountainCar(),
    DistractionWorldTest()
]

@pytest.mark.parametrize('test_case', TEST_CASES)
@pytest.mark.timeout(900)
def test_bsuite(
    test_case: BSuiteTestCase,
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
