import filelock
import pytest

from corerl.sql_logging.sql_logging import SQLEngineConfig, get_sql_engine
from test.behavior.bsuite import BSuiteTestCase
from test.behavior.calibration.cases import CalibrationTest
from test.behavior.distraction_world.cases import DistractionWorldTest
from test.behavior.mountain_car.cases import MountainCar, StandStillMountainCar
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
)
from test.behavior.t_maze.cases import LongTMazeTest, MediumTMazeTest, ShortTMazeTest
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
    DistractionWorldTest(),
    GoalSaturationTest(),
    MountainCar(),
    MultiActionSaturationTest(),
    SaturationTest(),
    StandStillMountainCar(),
    WindyRoomTest(),
    ExpandingBoundsSaturationTest(),
    SetpointChangeSaturationTest(),
    SlowExpandingBoundsSaturationTest(),
    DeltaChangeSaturationTest(),
    CalibrationTest(),
    MultiActionSaturationGreedificationTest(),
    MultiActionSaturationBadOfflineDataTest(),
    MultiActionSaturationGoodOfflineDataTest(),
    ShortTMazeTest(),
    MediumTMazeTest(),
    LongTMazeTest(),
    MCARSaturationEasyTest(),
    MCARSaturationMediumTest(),
    MCARSaturationHardTest(),
]

ZERO_ONE_FEATURES = [
    'base', # special feature indicating "no features enabled"
    'recency_bias_buffer',
    'regenerative_optimism',
]


def all_except(flags: list[str]):
    return {
        f: True for f in ZERO_ONE_FEATURES if f not in flags
    }


KNOWN_FAILURES: dict[str, bool | dict[str, bool]] = {
    WindyRoomTest.name: True,
}


def _zero_one_matrix(flags: list[str]):
    """
    Produces a matrix of feature flags with only one-hot values.
    Example:
    { 'delta_actions': True, 'zone_violations': False, 'action_embedding': False }
    """
    matrix: list[dict[str, bool]] = []
    for flag in flags:
        vals = { f: False for f in flags }
        vals[flag] = True
        matrix.append(vals)

    return matrix


@pytest.mark.parametrize('test_case', TEST_CASES, ids=lambda tc: tc.name)
@pytest.mark.parametrize('feature_flags', _zero_one_matrix(ZERO_ONE_FEATURES), ids=str)
def test_bsuite(
    test_case: BSuiteTestCase,
    bsuite_tsdb: None,
    feature_flags: dict[str, bool],
):
    # skip the test if any required feature is disabled
    for req_feature in test_case.required_features:
        if not feature_flags[req_feature]:
            pytest.skip()

    enabled_features = [
        feature for feature, enabled in feature_flags.items() if enabled
    ]
    feature_postfix = '_'.join(enabled_features)
    if feature_postfix:
        feature_postfix = '_' + feature_postfix

    # examples:
    # mountain_car_base  (no features enabled)
    # mountain_car_delta_actions_zone_violations  (two features enabled)
    schema = test_case.name.lower().replace(' ', '_') + feature_postfix


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
    metrics_table = test_case.execute_test(engine, db_name, schema, feature_flags)

    try:
        test_case.evaluate_outcomes(engine, metrics_table, feature_flags)
    except AssertionError as e:
        if test_case.name not in KNOWN_FAILURES:
            raise e

        # if all known to fail for all features, xfail
        fail_conditions = KNOWN_FAILURES[test_case.name]
        if fail_conditions is True:
            pytest.xfail()

        # if known to fail for given enabled features, xfail
        assert isinstance(fail_conditions, dict)
        if any(fail_conditions.get(f, False) for f in enabled_features):
            pytest.xfail()

        raise e
