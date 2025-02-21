import pytest
from sqlalchemy import Engine

from test.behavior.bsuite import BSuiteTestCase
from test.behavior.saturation.cases import (
    DelayedSaturationTest,
    DeltaSaturationTest,
    SaturationTest,
)
from test.behavior.stand_still_mountain_car.cases import StandStillMountainCar

TEST_CASES = [
    DelayedSaturationTest(),
    DeltaSaturationTest(),
    SaturationTest(),
    StandStillMountainCar(),
]

@pytest.mark.parametrize('test_case', TEST_CASES)
@pytest.mark.timeout(900)
def test_bsuite(
    test_case: BSuiteTestCase,
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
):
    port = tsdb_engine.url.port
    assert port is not None
    metrics_table = test_case.execute_test(tsdb_engine, port, tsdb_tmp_db_name)
    test_case.evaluate_outcomes(metrics_table)
