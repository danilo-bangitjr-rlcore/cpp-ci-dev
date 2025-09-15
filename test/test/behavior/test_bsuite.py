from typing import Any

import filelock
import pytest
from corerl.sql_logging.sql_logging import SQLEngineConfig
from lib_sql.engine import get_sql_engine

from test.behavior.bsuite import BehaviourCategory, BSuiteTestCase
from test.behavior.calibration.cases import CalibrationTest
from test.behavior.distraction_world.cases import DistractionWorldTest
from test.behavior.mountain_car.cases import MountainCar, StandStillMountainCar
from test.behavior.pertubation.cases import (
    MultiActionSaturationPerturbationTest,
    SaturationPerturbationTest,
    StandStillMountainCarPerturbationTest,
)
from test.behavior.saturation.cases import (
    DelayedSaturationTest,
    DeltaChangeSaturationTest,
    ExpandingBoundsSaturationTest,
    GoalSaturationTest,
    MCARSaturationEasyTest,
    MCARSaturationHardTest,
    MCARSaturationMediumTest,
    MultiActionSaturationBadOfflineDataTest,
    MultiActionSaturationGoodOfflineDataTest,
    MultiActionSaturationGreedificationTest,
    MultiActionSaturationTest,
    SaturationTest,
    SetpointChangeSaturationTest,
    SlowExpandingBoundsSaturationTest,
    StickyMCARSaturationTest,
)
from test.behavior.t_maze.cases import LongTMazeTest, MediumTMazeTest, ShortTMazeTest
from test.behavior.windy_room.cases import WindyRoomTest
from test.infrastructure.utils.tsdb import init_postgres_container


@pytest.fixture(scope='session')
def bsuite_tsdb():
    # file locking is necessary because we want to run these tests
    # in parallel using xdist, and we can't have a true session scope.
    # this prevents every test from independently trying to setup docker

    # Note: we aren't cleaning up after ourselves here, because the bsuite
    # tsdb docker container is intended to be long-lived
    with filelock.FileLock('output/bsuite.lock'):
        init_postgres_container(
            name='tsdb-bsuite',
            restart=False,
            port=22222,
        )



TEST_CASES = [
    CalibrationTest(),
    DelayedSaturationTest(),
    DeltaChangeSaturationTest(),
    DistractionWorldTest(),
    ExpandingBoundsSaturationTest(),
    GoalSaturationTest(),
    LongTMazeTest(),
    MCARSaturationEasyTest(),
    MCARSaturationMediumTest(),
    MCARSaturationHardTest(),
    MediumTMazeTest(),
    MountainCar(),
    MultiActionSaturationBadOfflineDataTest(),
    MultiActionSaturationGoodOfflineDataTest(),
    MultiActionSaturationGreedificationTest(),
    MultiActionSaturationPerturbationTest(),
    MultiActionSaturationTest(),
    SaturationPerturbationTest(),
    SaturationTest(),
    SetpointChangeSaturationTest(),
    ShortTMazeTest(),
    SlowExpandingBoundsSaturationTest(),
    StandStillMountainCar(),
    StandStillMountainCarPerturbationTest(),
    StickyMCARSaturationTest(),
    WindyRoomTest(),
]


KNOWN_FAILURES: dict[str, bool | dict[str, bool]] = {
    WindyRoomTest.name: True,
}


def get_tests_by_category(category: BehaviourCategory) -> list[BSuiteTestCase]:
    return [
        test_case for test_case in TEST_CASES
        if test_case.category == category
    ]


@pytest.fixture(scope='session')
def feature_flag(pytestconfig: Any):
    return pytestconfig.getoption('feature_flag')


@pytest.mark.parametrize('test_case', TEST_CASES, ids=lambda tc: tc.name)
def test_bsuite(
    test_case: BSuiteTestCase,
    bsuite_tsdb: None,
    feature_flag: str,
):
    # skip the test if any required feature is disabled
    if test_case.required_features and feature_flag not in test_case.required_features:
        pytest.skip()

    feature_flags = { feature_flag: True }
    schema = test_case.name.lower().replace(' ', '_') + f'_{feature_flag}'

    PORT = 22222
    db_name = 'bsuite'
    cfg = SQLEngineConfig(
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='workstation',
        port=PORT,
    )
    engine = get_sql_engine(cfg, db_name)
    metrics_table, runtime_info = test_case.execute_test(engine, db_name, schema, feature_flags)

    try:
        test_case.evaluate_outcomes(engine, metrics_table, feature_flags, runtime_info)
    except AssertionError as e:
        if test_case.name not in KNOWN_FAILURES:
            raise e

        # if all known to fail for all features, xfail
        fail_conditions = KNOWN_FAILURES[test_case.name]
        if fail_conditions is True:
            pytest.xfail()

        # if known to fail for given enabled features, xfail
        assert isinstance(fail_conditions, dict)
        if any(fail_conditions.get(f, False) for f, enabled in feature_flags.items() if enabled):
            pytest.xfail()

        raise e
