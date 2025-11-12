import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine

from test.behavior import utils
from test.behavior.bsuite import BehaviourCategory, BSuiteTestCase


class SaturationTest(BSuiteTestCase):
    name = 'saturation'
    config = 'test/behavior/saturation/config.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class GoalSaturationTest(BSuiteTestCase):
    name = 'saturation goal'
    config = 'test/behavior/saturation/saturation_goals.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    lower_bounds = { 'reward': -0.5}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }


class DelayedSaturationTest(BSuiteTestCase):
    name = 'delayed saturation'
    config = 'test/behavior/saturation/direct_delayed.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    lower_bounds = { 'reward': -0.19 }
    upper_warns = { 'avg_critic_loss': 0.002, 'actor_loss': -1.55 }

class MultiActionSaturationTest(BSuiteTestCase):
    name = 'multi action saturation'
    config = 'test/behavior/saturation/multi_action_config.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    lower_bounds = { 'reward': -0.1}

class ExpandingBoundsSaturationTest(BSuiteTestCase):
    name = 'expanding bounds saturation'
    config = 'test/behavior/saturation/expanding_bounds.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.085}

class SlowExpandingBoundsSaturationTest(BSuiteTestCase):
    name = 'slow expanding bounds saturation'
    config = 'test/behavior/saturation/slow_expanding_bounds.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.085}

class SetpointChangeSaturationTest(BSuiteTestCase):
    name = 'setpoint change saturation'
    config = 'test/behavior/saturation/setpoint_change.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.085}

class DeltaChangeSaturationTest(BSuiteTestCase):
    name = 'delta change saturation'
    config = 'test/behavior/saturation/changing_delta.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.085}

class MCARSaturationEasyTest(BSuiteTestCase):
    name = 'easy missing at random saturation'
    config = 'test/behavior/saturation/mcar_saturation_easy.yaml'
    category = {BehaviourCategory.ROBUSTNESS}
    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }

class MCARSaturationMediumTest(BSuiteTestCase):
    name = 'medium missing at random saturation'
    config = 'test/behavior/saturation/mcar_saturation_medium.yaml'
    category = {BehaviourCategory.ROBUSTNESS}
    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }

class MCARSaturationHardTest(BSuiteTestCase):
    name = 'hard missing at random saturation'
    config = 'test/behavior/saturation/mcar_saturation_hard.yaml'
    category = {BehaviourCategory.ROBUSTNESS}
    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }

class StickyMCARSaturationTest(BSuiteTestCase):
    name = 'sticky missing at random saturation'
    config = 'test/behavior/saturation/sticky_mcar_saturation.yaml'
    category = {BehaviourCategory.ROBUSTNESS}
    lower_bounds = { 'reward': -0.085}
    upper_warns = { 'avg_critic_loss': 0.003, 'actor_loss': -1.55 }

class MultiActionSaturationGreedificationTest(BSuiteTestCase):
    """
    Test how quickly the actor greedifies with respect to the critic
    when the actor and critic are separately pretrained to prefer different modes.
    """
    name = 'multi action saturation greedification'
    config = 'test/behavior/saturation/multi_action_greedification.yaml'
    setup_cfgs = ['test/behavior/saturation/multi_action_greedification_actor_pretrain.yaml',
                  'test/behavior/saturation/multi_action_greedification_critic_pretrain.yaml']

    upper_bounds = { 'greed_dist_online': 0.1}
    category = {BehaviourCategory.GREEDY}
    def setup(self, engine: Engine, infra_overrides: dict[str, object], feature_overrides: dict[str, bool]):
        """
        First Run: Pretrain actor on setpoints of [0.2, 0.8, 0.2]
        Second Run: Pretrain critic on setpoints of [0.8, 0.2, 0.8]
        Third Run: Take actor from first run and critic from second run
        and conduct a new run with setpoints of [0.8, 0.2, 0.8]
        """
        save_path = Path("outputs/greedification_test")
        checkpoint_path = save_path / "checkpoints"

        case_overrides = {
            'metrics.enabled': False,
            'evals.enabled': False,
        }

        overrides = self._overrides | infra_overrides | feature_overrides | case_overrides

        parts = [f'{k}={v}' for k, v in overrides.items()]

        # Actor Pretraining Run
        proc = subprocess.run([
            'python', 'main.py',
            '--base', '.',
            '--config-name', self.setup_cfgs[0],
            *parts,
        ], check=False)
        proc.check_returncode()

        actor_checkpoints = list(checkpoint_path.glob('*'))
        actor_checkpoint: Path = sorted(actor_checkpoints)[-1]
        actor_net_path = actor_checkpoint / 'actor' / 'actor_net'
        actor_opt_path = actor_checkpoint / 'actor' / 'actor_opt'
        sampler_net_path = actor_checkpoint / 'actor' / 'sampler_net'
        sampler_opt_path = actor_checkpoint / 'actor' / 'sampler_opt'

        # Critic Pretraining Run
        proc = subprocess.run([
            'python', 'main.py',
            '--base', '.',
            '--config-name', self.setup_cfgs[1],
            *parts,
        ], check=False)
        proc.check_returncode()

        critic_checkpoints = list(checkpoint_path.glob('*'))
        critic_checkpoint: Path = sorted(critic_checkpoints)[-1]
        critic_net_path = critic_checkpoint / 'q_critic' / 'critic_net'
        critic_opt_path = critic_checkpoint / 'q_critic' / 'critic_opt_0'
        target_net_path = critic_checkpoint / 'q_critic' / 'critic_target'

        # Create new checkpoint directory that will store the pretrained actor and critic
        now = datetime.now(UTC)
        new_checkpoint_path = checkpoint_path / f'{str(now).replace(':', '_')}'
        new_checkpoint_path.mkdir(exist_ok=True, parents=True)
        actor_destination = new_checkpoint_path / "actor"
        actor_destination.mkdir(exist_ok=True, parents=True)
        critic_destination = new_checkpoint_path / "q_critic"
        critic_destination.mkdir(exist_ok=True, parents=True)

        # Move policy files to new actor directory
        actor_net_path.rename(actor_destination / actor_net_path.name)
        actor_opt_path.rename(actor_destination / actor_opt_path.name)
        sampler_net_path.rename(actor_destination / sampler_net_path.name)
        sampler_opt_path.rename(actor_destination / sampler_opt_path.name)

        # Move critic files to new critic directory
        critic_net_path.rename(critic_destination / critic_net_path.name)
        critic_opt_path.rename(critic_destination / critic_opt_path.name)
        target_net_path.rename(critic_destination / target_net_path.name)

class MultiActionSaturationGoodOfflineDataTest(BSuiteTestCase):
    """
    Test whether the agent quickly converges to good performance
    when the replay buffer is preloaded with good offline data
    """
    name = 'multi action saturation good offline data'
    config = 'test/behavior/saturation/multi_action_good_offline_data.yaml'
    category = {BehaviourCategory.PERFORMANCE}
    lower_bounds = { 'reward': -0.1}

    def setup(self, engine: Engine, infra_overrides: dict[str, object], feature_overrides: dict[str, bool]):
        # Read offline data from csv
        obs_path = Path('test/behavior/saturation/good_offline_data.csv')
        df = utils.read_offline_data(obs_path)
        sql_tups = []
        for col_name in df.columns:
            sql_tups += utils.column_to_sql_tups(df[col_name])

        # Write offline data to db
        data_writer = utils.get_offline_data_writer(engine, infra_overrides)
        for sql_tup in sql_tups:
            data_writer.write(timestamp=sql_tup[0], name=sql_tup[2], val=sql_tup[1])

        data_writer.close()

class MultiActionSaturationBadOfflineDataTest(BSuiteTestCase):
    """
    Test whether the agent can recover and learn a good policy when its replay buffer is preloaded with data produced
    under different environment dynamics than the environment that it is evaluated in.

    The offline dataset (bad_offline_data.csv) was produced by deploying a GAC agent in MultiActionSaturation
    with the following configuration for 2000 steps:
        frequencies: [4.0, 10.0, 20.0]
        phase_shifts: [pi/2, 0.0, pi/4]

    In this test, our agent is evaluated for 2000 steps in MultiActionSaturation with the following configuration:
        frequencies: [1.0, 1.5, 2.0]
        phase_shifts: [0.0, pi/4, pi/2]
    """
    name = 'multi action saturation bad offline data'
    config = 'test/behavior/saturation/multi_action_bad_offline_data.yaml'
    category = {BehaviourCategory.NONSTATIONARY}
    lower_bounds = { 'reward': -0.1}

    def setup(self, engine: Engine, infra_overrides: dict[str, object], feature_overrides: dict[str, bool]):
        # Read offline data from csv
        obs_path = Path('test/behavior/saturation/bad_offline_data.csv')
        df = utils.read_offline_data(obs_path)
        sql_tups = []
        for col_name in df.columns:
            sql_tups += utils.column_to_sql_tups(df[col_name])

        # Write offline data to db
        data_writer = utils.get_offline_data_writer(engine, infra_overrides)

        # ensure table exists by writing and syncing a dummy value
        data_writer.write(timestamp=datetime.now(UTC), name="dummy", val=0.0)
        data_writer.flush()

        for sql_tup in sql_tups:
            data_writer.write(timestamp=sql_tup[0], name=sql_tup[2], val=sql_tup[1])
        data_writer.close()

class SingleActionMultimodalSaturationTest(BSuiteTestCase):
    """
    Test whether the agent can converge to the optimal policy when Saturation has a multimodal reward function.
    Single action dimension.
    """
    name = 'single action multimodal saturation'
    config = 'test/behavior/saturation/single_action_multimodal_saturation.yaml'
    category = {BehaviourCategory.EXPLORATION}
    lower_bounds = { 'reward': 0.302 } # Just higher than the second best local optimum

class MultiActionMultimodalSaturationTest(BSuiteTestCase):
    """
    Test whether the agent can converge to the optimal policy when Saturation has a multimodal reward function.
    Action dimension = 3.
    """
    name = 'multi action multimodal saturation'
    config = 'test/behavior/saturation/multi_action_multimodal_saturation.yaml'
    category = {BehaviourCategory.EXPLORATION}
    lower_bounds = { 'reward': 0.906 } # Just higher than the second best local optimum

class SingleActionDelayedObsSaturationTest(BSuiteTestCase):
    """
    Version of Saturation where the observed saturation is the amount that exceeds the filter threshold.
    The observation is also delayed by a configurable number of steps.
    Action dimension = 1.
    """
    name = 'single action delayed obs saturation'
    config = 'test/behavior/saturation/single_action_delayed_obs_saturation.yaml'
    category = {BehaviourCategory.CREDIT_ASSIGNMENT}
    lower_bounds = { 'reward': -0.04 }

class MultiActionDelayedObsSaturationTest(BSuiteTestCase):
    """
    Version of Saturation where the observed saturation is the amount that exceeds the filter threshold.
    The observation is also delayed by a configurable number of steps.
    Action dimension = 3.
    """
    name = 'multi action delayed obs saturation'
    config = 'test/behavior/saturation/multi_action_delayed_obs_saturation.yaml'
    category = {BehaviourCategory.CREDIT_ASSIGNMENT}
    lower_bounds = { 'reward': -0.15 }

class SingleActionFilteredSaturationTest(BSuiteTestCase):
    """
    Version of Saturation where the observed saturation is the amount that exceeds the filter threshold.
    Tests whether the agent can converge to the optimal policy when there is a region of 0 reward.
    Action dimension = 1.
    """
    name = 'single action filtered saturation'
    config = 'test/behavior/saturation/single_action_filtered_saturation.yaml'
    category = {BehaviourCategory.EXPLORATION}
    lower_bounds = { 'reward': -0.04 }

class MultiActionFilteredSaturationTest(BSuiteTestCase):
    """
    Version of Saturation where the observed saturation is the amount that exceeds the filter threshold.
    Tests whether the agent can converge to the optimal policy when there is a region of 0 reward.
    Action dimension = 3.
    """
    name = 'multi action filtered saturation'
    config = 'test/behavior/saturation/multi_action_filtered_saturation.yaml'
    category = {BehaviourCategory.EXPLORATION}
    lower_bounds = { 'reward': -0.15 }
